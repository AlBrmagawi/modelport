import hashlib
import socket
from contextlib import contextmanager

import httpcore
import pytest

from modelport.config import Settings
from modelport.domain import HTTPSImport, HubImport, RemoteFile
from modelport.errors import ModelPortError
from modelport.ports import ExecutionContext
from modelport.remote import (
    PublicBackend,
    download_file,
    hub_request,
    public_address,
    resolve_public,
    validate_files,
    validate_url,
)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.1.2.3",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "224.0.0.1",
        "::1",
        "fe80::1",
        "fc00::1",
        "::ffff:127.0.0.1",
        "64:ff9b::7f00:1",
        "2002:7f00:1::",
        "2001:db8::1",
    ],
)
def test_nonpublic_and_translation_addresses_rejected(address):
    assert not public_address(address)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/a.onnx",
        "file:///etc/passwd",
        "https://user:pass@example.com/x",
        "https://127.0.0.1/x",
        "https://[::1]/x",
        "https://example.com:8443/x",
        "https://metadata.google.internal/x",
        "https://example.com/x#fragment",
        "https://example.com\\@127.0.0.1/x",
        "https://example.com/\r\nx",
    ],
)
def test_unsafe_urls(url):
    with pytest.raises(ModelPortError):
        validate_url(url)


def test_mixed_dns_answers_and_connect_time_rebinding_fail_closed(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET, 1, 6, "", ("93.184.216.34", 443)),
            (socket.AF_INET, 1, 6, "", ("127.0.0.1", 443)),
        ],
    )
    with pytest.raises(ModelPortError, match="Every DNS"):
        resolve_public("example.com", 443)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, 1, 6, "", ("169.254.169.254", 443))],
    )
    with pytest.raises(ModelPortError):
        PublicBackend().connect_tcp("example.com", 443)


def test_backend_pins_numeric_address_and_verifies_actual_peer(monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, 1, 6, "", ("93.184.216.34", 443))]
    )

    class Stream:
        closed = False
        peer = ("93.184.216.34", 443)

        def get_extra_info(self, name):
            return self.peer

        def close(self):
            self.closed = True

    stream = Stream()
    connected = []

    def connect(self, host, port, *args):
        connected.append((host, port))
        return stream

    monkeypatch.setattr(httpcore.SyncBackend, "connect_tcp", connect)
    assert PublicBackend().connect_tcp("example.com", 443) is stream
    assert connected == [("93.184.216.34", 443)]
    stream.peer = ("127.0.0.1", 443)
    with pytest.raises(ModelPortError, match="actual socket peer"):
        PublicBackend().connect_tcp("example.com", 443)
    assert stream.closed


def fake_pool(monkeypatch, responses):
    class Pool:
        def __init__(self, **options):
            assert isinstance(options["network_backend"], PublicBackend)
            assert "proxy" not in options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        @contextmanager
        def stream(self, method, url, headers, extensions):
            assert "Authorization" not in headers and "Cookie" not in headers
            response = responses.pop(0)
            yield response

    monkeypatch.setattr(httpcore, "ConnectionPool", Pool)


def context(tmp_path):
    return ExecutionContext(tmp_path, Settings(data_dir=tmp_path), lambda *args: None, lambda: None)


def test_redirect_to_private_destination_is_rejected(monkeypatch, tmp_path):
    fake_pool(
        monkeypatch,
        [
            httpcore.Response(
                302, headers={b"Location": b"https://169.254.169.254/latest/meta-data"}
            )
        ],
    )
    with pytest.raises(ModelPortError, match="public HTTPS"):
        download_file(
            RemoteFile(url="https://example.com/a.onnx", path="a.onnx", sha256="a" * 64),
            tmp_path / "out",
            100,
            context(tmp_path),
        )
    assert not (tmp_path / "out").exists()


def test_stream_size_and_checksum_enforced(monkeypatch, tmp_path):
    item = RemoteFile(
        url="https://example.com/a.onnx", path="a.onnx", sha256=hashlib.sha256(b"real").hexdigest()
    )
    fake_pool(monkeypatch, [httpcore.Response(200, content=b"wrong")])
    with pytest.raises(ModelPortError, match="SHA-256"):
        download_file(item, tmp_path / "wrong", 100, context(tmp_path))
    fake_pool(monkeypatch, [httpcore.Response(200, content=b"123456")])
    with pytest.raises(ModelPortError, match="byte budget"):
        download_file(item, tmp_path / "large", 4, context(tmp_path))
    fake_pool(monkeypatch, [httpcore.Response(200, content=b"real")])
    evidence = download_file(item, tmp_path / "right", 100, context(tmp_path))
    assert evidence["sha256"] == item.sha256 and (tmp_path / "right").read_bytes() == b"real"


def test_hub_requires_full_revision_and_rejects_unsafe_files():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        HubImport(
            repository="owner/model",
            revision="main",
            files=[{"path": "model.onnx", "sha256": "a" * 64}],
        )
    request = hub_request(
        HubImport(
            repository="owner/model",
            revision="1" * 40,
            files=[{"path": "model.onnx", "sha256": "a" * 64}],
        )
    )
    assert "/resolve/" + "1" * 40 + "/model.onnx" in request.files[0].url
    for filename in ("../escape.onnx", "model.pt", "code.py", "MODEL.onnx:stream"):
        with pytest.raises(ModelPortError):
            validate_files(
                HTTPSImport(
                    files=[
                        RemoteFile(url="https://example.com/model", path=filename, sha256="a" * 64)
                    ]
                )
            )
