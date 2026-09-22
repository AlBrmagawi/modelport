"""Check distributable contents, source correspondence and release checksums."""

import base64
import csv
import hashlib
import io
import json
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PARTS = {
    ".git",
    ".venv",
    ".tools",
    ".modelport",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "test-results",
    "playwright-report",
    "demo-results",
    "demo-dual-results",
    "container-job",
}


def check_name(name: str) -> None:
    path = PurePosixPath(name)
    assert not path.is_absolute() and ".." not in path.parts, name
    assert "\\" not in name and ":" not in name, name
    assert not PRIVATE_PARTS.intersection(path.parts), name
    assert not path.name.startswith(".env") or path.name == ".env.example", name
    assert path.name != "operator-token" and not path.name.startswith("modelport.db"), name
    assert path.suffix not in {".pyc", ".pem", ".key", ".onnx", ".safetensors", ".npz"}, name


def read_zip(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert len(names) == len({name.casefold() for name in names}), "Duplicate archive paths"
        for name in names:
            check_name(name)
        assert archive.testzip() is None, "Corrupt ZIP member"
        return {
            entry.filename: archive.read(entry)
            for entry in archive.infolist()
            if not entry.is_dir()
        }


def main() -> None:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    directory = ROOT / "release" / version
    prefix = f"modelport-{version}"
    wheel_name = f"{prefix}-py3-none-any.whl"
    expected = {
        wheel_name,
        f"{prefix}.tar.gz",
        f"{prefix}-source.zip",
        "dependency-inventory.json",
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY_NOTICES.md",
        "RELEASE_NOTES.md",
    }
    checksums = {}
    for line in (directory / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert name in expected and name not in checksums, name
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest, name
        checksums[name] = digest
    assert set(checksums) == expected, "Incomplete checksum manifest"

    source = read_zip(directory / f"{prefix}-source.zip")
    with tarfile.open(directory / f"{prefix}.tar.gz") as archive:
        members = archive.getmembers()
        assert len(members) == len({member.name.casefold() for member in members})
        tar_files = {}
        for member in members:
            check_name(member.name)
            assert member.isfile() or member.isdir(), "Source archive contains a special file"
            if member.isfile():
                stream = archive.extractfile(member)
                assert stream is not None
                with stream:
                    tar_files[member.name] = stream.read()
    assert tar_files == source, "ZIP and tarball differ"
    assert all(name.startswith(prefix + "/") for name in source)
    required = {
        "README.md",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        "SUPPORT.md",
        "SECURITY.md",
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY_NOTICES.md",
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        ".gitignore",
        ".gitattributes",
        ".editorconfig",
        ".gitleaks.toml",
        ".dockerignore",
        ".env.example",
        ".github/workflows/ci.yml",
        "docker/Dockerfile",
        "docker/compose.yml",
        "docker/compose.isolated.yml",
        "start-modelport.cmd",
        "scripts/start-docker.ps1",
        "scripts/check.py",
        "scripts/verify_release.py",
        "docs/README.md",
        "docs/getting-started.md",
        "docs/dependencies.md",
        "docs/assets/modelport-banner.svg",
        "tests/conftest.py",
        "web/package-lock.json",
        "web/.prettierignore",
        "web/dist/index.html",
        "web/dist/third-party-licenses.txt",
    }
    assert {prefix + "/" + name for name in required} <= source.keys(), "Missing source files"
    for name, content in source.items():
        relative = name.removeprefix(prefix + "/")
        if relative == "PKG-INFO":
            continue
        assert (ROOT / relative).is_file(), name
        assert (ROOT / relative).read_bytes() == content, f"Source differs: {relative}"

    wheel = read_zip(directory / wheel_name)
    for path in (ROOT / "src/modelport").rglob("*.py"):
        relative = path.relative_to(ROOT / "src").as_posix()
        assert wheel.get(relative) == path.read_bytes(), f"Wheel code differs: {relative}"
    for path in (ROOT / "web/dist").rglob("*"):
        if path.is_file():
            relative = path.relative_to(ROOT / "web/dist").as_posix()
            assert wheel.get("modelport/web/" + relative) == path.read_bytes(), relative
    info = f"modelport-{version}.dist-info"
    metadata = BytesParser().parsebytes(wheel[info + "/METADATA"])
    assert metadata["Name"] == "modelport" and metadata["Version"] == version
    assert metadata["License-Expression"] == "Apache-2.0"
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
        assert wheel[info + "/licenses/" + name] == (ROOT / name).read_bytes(), name
    records = list(csv.reader(io.StringIO(wheel[info + "/RECORD"].decode())))
    assert len(records) == len(wheel) and {row[0] for row in records} == wheel.keys()
    for name, digest, size in records:
        if name == info + "/RECORD":
            assert digest == size == ""
            continue
        actual = base64.urlsafe_b64encode(hashlib.sha256(wheel[name]).digest()).rstrip(b"=")
        assert digest == "sha256=" + actual.decode() and int(size) == len(wheel[name]), name

    evidence = {
        "version": version,
        "checksums_verified": len(checksums),
        "source_entries": len(source),
        "wheel_entries": len(wheel),
        "source_and_dashboard_match": True,
        "private_files_absent": True,
        "licenses_and_metadata_verified": True,
        "wheel_sha256": checksums[wheel_name],
    }
    (ROOT / ".tools").mkdir(exist_ok=True)
    (ROOT / ".tools/release-verification.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
