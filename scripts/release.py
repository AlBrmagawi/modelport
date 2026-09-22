"""Build local release artifacts, inventories and checksums. Never publishes."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import tarfile
import tomllib
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-frontend", action="store_true", help="Use the existing web/dist")
    args = parser.parse_args()
    uv = shutil.which("uv") or str(ROOT / ".tools" / "uv" / "uv.exe")
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    if not args.skip_frontend:
        if not npm:
            raise SystemExit("Node.js/npm is required to build the release dashboard")
        subprocess.run([npm, "ci"], cwd=ROOT / "web", check=True)
        subprocess.run([npm, "run", "build"], cwd=ROOT / "web", check=True)
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    directory = ROOT / "release" / version
    directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [uv, "build", "--out-dir", str(directory)],
        cwd=ROOT,
        check=True,
        env={**os.environ, "MODELPORT_REQUIRE_WEB": "1"},
    )
    sdist = directory / f"modelport-{version}.tar.gz"
    wheel = directory / f"modelport-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as archive:
        assert "modelport/web/index.html" in archive.namelist()
        assert "modelport/web/third-party-licenses.txt" in archive.namelist()
    source = directory / f"modelport-{version}-source.zip"
    with (
        tarfile.open(sdist) as archive,
        zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as output,
    ):
        for member in archive.getmembers():
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if any(
                part in {".tools", ".venv", ".modelport", "node_modules", ".env"} for part in parts
            ):
                raise RuntimeError(f"Private build content found: {member.name}")
            stream = archive.extractfile(member)
            assert stream is not None
            with stream:
                output.writestr(member.name, stream.read())
    python = []
    for distribution in importlib.metadata.distributions():
        metadata = distribution.metadata
        python.append(
            {
                "name": metadata["Name"],
                "version": distribution.version,
                "reported_license": metadata.get("License-Expression") or metadata.get("License"),
            }
        )
    lock = json.loads((ROOT / "web" / "package-lock.json").read_text())
    inventory = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "Build environment including development dependencies; npm entries are locked",
        "python": sorted(python, key=lambda item: item["name"].lower()),
        "javascript": [
            {
                "path": name,
                **{
                    key: value
                    for key, value in entry.items()
                    if key in {"version", "license", "integrity", "dev"}
                },
            }
            for name, entry in lock["packages"].items()
            if name
        ],
    }
    inventory_path = directory / "dependency-inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    for filename in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
        shutil.copyfile(ROOT / filename, directory / filename)
    shutil.copyfile(ROOT / "docs" / "release-notes.md", directory / "RELEASE_NOTES.md")
    artifacts = [
        wheel,
        sdist,
        source,
        inventory_path,
        directory / "LICENSE",
        directory / "NOTICE",
        directory / "THIRD_PARTY_NOTICES.md",
        directory / "RELEASE_NOTES.md",
    ]
    (directory / "SHA256SUMS.txt").write_text(
        "".join(
            f"{hashlib.sha256(file.read_bytes()).hexdigest()}  {file.name}\n" for file in artifacts
        ),
        encoding="utf-8",
    )
    print(f"Release artifacts ready in {directory}; nothing was published.")


if __name__ == "__main__":
    main()
