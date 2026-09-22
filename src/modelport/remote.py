"""Public HTTPS downloads with destination pinning and no ambient credentials/proxies."""

import hashlib
import ipaddress
import socket
import time
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit

import httpcore

from modelport.domain import HTTPSImport, HubImport, RemoteFile
from modelport.errors import ModelPortError
from modelport.ports import ExecutionContext
from modelport.security import confined, safe_relative


def public_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if not address.is_global or address.is_multicast or address.is_reserved:
        return False
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped or address.sixtofour or address.teredo:
            return False
        if address in ipaddress.ip_network("64:ff9b::/96") or address in ipaddress.ip_network(
            "64:ff9b:1::/48"
        ):
            return False
    return True


def validate_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
        if (
            len(url) > 8192
            or any(ord(c) < 33 or ord(c) > 126 for c in url)
            or "\\" in url
            or parsed.scheme != "https"
            or not parsed.hostname
            or parsed.port not in {None, 443}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or "%" in parsed.hostname
        ):
            raise ValueError()
        host = parsed.hostname.encode("idna").decode("ascii")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if "." not in host or host.endswith((".localhost", ".local", ".internal")):
                raise ValueError() from None
        else:
            if not public_address(str(address)):
                raise ValueError()
        return url
    except (ValueError, UnicodeError) as exc:
        raise ModelPortError(
            "REMOTE_ADDRESS_BLOCKED",
            "Use a public HTTPS URL on port 443 without credentials or fragments",
        ) from exc


def resolve_public(host: str, port: int) -> list[str]:
    if port != 443:
        raise ModelPortError("REMOTE_ADDRESS_BLOCKED", "Only HTTPS port 443 is permitted")
    try:
        values = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ModelPortError(
            "REMOTE_DNS_FAILED", "The public download host could not be resolved"
        ) from exc
    addresses = sorted({str(item[4][0]) for item in values})
    if not addresses or any(not public_address(address) for address in addresses):
        raise ModelPortError(
            "REMOTE_ADDRESS_BLOCKED", "Every DNS answer must be a public unicast address"
        )
    return addresses


class PublicBackend(httpcore.SyncBackend):
    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        addresses = resolve_public(host, port)
        last_error = None
        for address in addresses[:4]:
            try:
                stream = super().connect_tcp(address, port, timeout, None, socket_options)
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last_error = exc
                continue
            peer = stream.get_extra_info("server_addr")
            if (
                not peer
                or ipaddress.ip_address(peer[0]) != ipaddress.ip_address(address)
                or peer[1] != port
            ):
                stream.close()
                raise ModelPortError(
                    "REMOTE_ADDRESS_BLOCKED",
                    "The actual socket peer differs from its pinned public address",
                )
            return stream
        raise ModelPortError(
            "REMOTE_CONNECT_FAILED", "Could not connect to a pinned public address"
        ) from last_error

    def connect_unix_socket(self, *args, **kwargs):
        raise ModelPortError("REMOTE_ADDRESS_BLOCKED", "Local sockets are disabled")


def download_file(
    item: RemoteFile, destination: Path, limit: int, context: ExecutionContext
) -> dict:
    url = validate_url(item.url)
    # The hostname remains the TLS SNI and certificate identity. Only connect_tcp is pinned.
    with httpcore.ConnectionPool(
        network_backend=PublicBackend(), max_connections=1, max_keepalive_connections=0, retries=0
    ) as pool:
        for hop in range(4):
            context.check_deadline()
            validate_url(url)
            try:
                with pool.stream(
                    "GET",
                    url,
                    headers={
                        "User-Agent": "ModelPort/0.1",
                        "Accept-Encoding": "identity",
                        "Accept": "application/octet-stream",
                    },
                    extensions={"timeout": {"connect": 10, "read": 10, "write": 10, "pool": 10}},
                ) as response:
                    headers = {}
                    for key, value in response.headers:
                        key = key.lower()
                        if key in headers and key in {
                            b"content-length",
                            b"location",
                            b"content-encoding",
                        }:
                            raise ModelPortError(
                                "REMOTE_PROTOCOL_ERROR", "Ambiguous remote response headers"
                            )
                        headers[key] = value
                    if response.status in {301, 302, 303, 307, 308}:
                        if hop == 3 or b"location" not in headers:
                            raise ModelPortError(
                                "REMOTE_REDIRECT_LIMIT", "Download exceeded three redirects"
                            )
                        url = validate_url(urljoin(url, headers[b"location"].decode("ascii")))
                        continue
                    if response.status != 200:
                        raise ModelPortError(
                            "REMOTE_HTTP_ERROR", f"Public download returned HTTP {response.status}"
                        )
                    if headers.get(b"content-encoding", b"identity").lower() != b"identity":
                        raise ModelPortError(
                            "REMOTE_PROTOCOL_ERROR", "Compressed HTTP content is disabled"
                        )
                    try:
                        declared = (
                            int(headers[b"content-length"])
                            if b"content-length" in headers
                            else None
                        )
                    except ValueError as exc:
                        raise ModelPortError(
                            "REMOTE_PROTOCOL_ERROR", "Invalid remote content length"
                        ) from exc
                    if declared is not None and not 0 < declared <= limit:
                        raise ModelPortError(
                            "RESOURCE_EXHAUSTED", "Remote file exceeds the remaining byte budget"
                        )
                    size, digest = 0, hashlib.sha256()
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("xb") as output:
                        for chunk in response.iter_stream():
                            context.check_deadline()
                            size += len(chunk)
                            if size > limit:
                                raise ModelPortError(
                                    "RESOURCE_EXHAUSTED", "Remote stream exceeds the byte budget"
                                )
                            output.write(chunk)
                            digest.update(chunk)
                    if not size or (declared is not None and size != declared):
                        raise ModelPortError(
                            "REMOTE_PROTOCOL_ERROR", "Remote file was empty or truncated"
                        )
                    if digest.hexdigest() != item.sha256:
                        raise ModelPortError(
                            "CHECKSUM_MISMATCH",
                            "Downloaded bytes do not match the required SHA-256",
                        )
                    return {
                        "url": item.url.split("?", 1)[0],
                        "path": item.path,
                        "sha256": item.sha256,
                        "size_bytes": size,
                        "redirects": hop,
                    }
            except (
                httpcore.NetworkError,
                httpcore.ProtocolError,
                httpcore.TimeoutException,
                UnicodeError,
            ) as exc:
                raise ModelPortError(
                    "REMOTE_DOWNLOAD_FAILED",
                    "HTTPS verification or bounded network transfer failed",
                ) from exc
    raise ModelPortError("REMOTE_REDIRECT_LIMIT", "Download exceeded redirect limit")


def validate_files(request: HTTPSImport) -> None:
    seen = set()
    for item in request.files:
        safe_relative(item.path)
        validate_url(item.url)
        if urlsplit(item.url).query:
            raise ModelPortError(
                "REMOTE_ADDRESS_BLOCKED",
                "Initial download URLs must be public unsigned URLs; "
                "authenticated imports are not supported",
            )
        if Path(item.path).suffix.lower() not in {".onnx", ".json", ".safetensors", ".data"}:
            raise ModelPortError(
                "UNSUPPORTED_FORMAT", "Remote imports accept documented model bundles only"
            )
        if item.path.casefold() in seen:
            raise ModelPortError("UNSAFE_PATH", "Remote destination filenames must be unique")
        seen.add(item.path.casefold())


def hub_request(request: HubImport) -> HTTPSImport:
    return HTTPSImport(
        files=[
            RemoteFile(
                url=f"https://huggingface.co/{request.repository}/resolve/{request.revision}/{quote(safe_relative(item.path))}",
                path=item.destination or item.path,
                sha256=item.sha256,
            )
            for item in request.files
        ]
    )


def download_bundle(request: HTTPSImport, directory: Path, context: ExecutionContext) -> dict:
    validate_files(request)
    directory.mkdir(mode=0o700)
    total, evidence, started = 0, [], time.monotonic()
    for item in request.files:
        context.stage("downloading", file=item.path)
        target = confined(directory, item.path, must_exist=False)
        record = download_file(
            item,
            target,
            min(context.settings.max_file_bytes, context.settings.max_bundle_bytes - total),
            context,
        )
        total += record["size_bytes"]
        evidence.append(record)
    return {
        "kind": "public-https",
        "files": evidence,
        "duration_seconds": time.monotonic() - started,
        "credentials": "none",
        "proxy": "disabled",
        "destination_policy": "public-only-pinned-v1",
    }
