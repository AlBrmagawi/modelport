import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from modelport.config import Settings
from modelport.errors import ModelPortError
from modelport.security import confined, digest_file, identity, reject_link, safe_relative


class ArtifactStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.data_dir

    def stage_import(self, source: Path) -> tuple[str, str]:
        source = source.absolute()
        if not source.exists():
            raise ModelPortError("NOT_FOUND", "Import source does not exist", status=404)
        # Reject links in every ancestor, including junctions above the selected root.
        for ancestor in [source, *source.parents]:
            reject_link(ancestor)
        staging_id = uuid.uuid4().hex
        target = self.root / "staging" / staging_id
        target.mkdir(mode=0o700)
        files = []
        if source.is_dir():
            for base, dirs, names in os.walk(source, followlinks=False):
                for name in dirs:
                    reject_link(Path(base) / name)
                for name in names:
                    files.append(Path(base) / name)
                    if len(files) > self.settings.max_files:
                        raise ModelPortError("RESOURCE_EXHAUSTED", "Bundle has too many files")
        else:
            files = [source]
        seen: set[str] = set()
        total = 0
        try:
            for file in files:
                reject_link(file)
                relative = file.relative_to(source).as_posix() if source.is_dir() else file.name
                safe_relative(relative)
                if relative.casefold() in seen:
                    raise ModelPortError("UNSAFE_PATH", "Case-colliding filenames are ambiguous")
                seen.add(relative.casefold())
                size = file.stat().st_size
                total += size
                if size > self.settings.max_file_bytes or total > self.settings.max_bundle_bytes:
                    raise ModelPortError(
                        "RESOURCE_EXHAUSTED", "Import exceeds configured byte limits"
                    )
                if file.suffix.lower() not in {".onnx", ".safetensors", ".json", ".data"}:
                    raise ModelPortError(
                        "UNSUPPORTED_FORMAT",
                        (
                            "Only ONNX, SafeTensors and documented bundles are accepted; "
                            "pickle, code"
                            " and archives are rejected"
                        ),
                    )
                dest = confined(target, relative, must_exist=False)
                dest.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                with file.open("rb") as inp, dest.open("xb") as out:
                    first = inp.read(1024)
                    if first.startswith(b"version https://git-lfs.github.com/spec/"):
                        raise ModelPortError(
                            "LFS_POINTER", "Download actual model bytes, not a Git LFS pointer"
                        )
                    out.write(first)
                    written += len(first)
                    for chunk in iter(lambda: inp.read(1024**2), b""):
                        written += len(chunk)
                        if written > size:
                            raise ModelPortError("SOURCE_CHANGED", "File changed while importing")
                        out.write(chunk)
                if written != size:
                    raise ModelPortError("SOURCE_CHANGED", "File changed while importing")
            if not files:
                raise ModelPortError("UNSUPPORTED_FORMAT", "The selected directory is empty")
        except Exception:
            shutil.rmtree(target)
            raise
        return staging_id, source.name

    def inventory(self, directory: Path) -> tuple[str, list[dict[str, Any]]]:
        files = []
        total = 0
        for path in sorted(directory.rglob("*")):
            reject_link(path)
            if path.is_file():
                relative = path.relative_to(directory).as_posix()
                safe_relative(relative)
                size = path.stat().st_size
                total += size
                if size > self.settings.max_file_bytes or total > self.settings.max_bundle_bytes:
                    raise ModelPortError("RESOURCE_EXHAUSTED", "Artifact exceeds byte limits")
                files.append({"path": relative, "size_bytes": size, "sha256": digest_file(path)})
                if len(files) > self.settings.max_files:
                    raise ModelPortError("RESOURCE_EXHAUSTED", "Artifact exceeds file count limit")
        if not files:
            raise ModelPortError("INVALID_MODEL", "Empty artifact cannot be published")
        return identity(files), files

    def path(self, artifact_id: str) -> Path:
        if len(artifact_id) != 64 or any(x not in "0123456789abcdef" for x in artifact_id):
            raise ModelPortError("NOT_FOUND", "Invalid artifact identifier", status=404)
        return self.root / "blobs" / artifact_id

    def publish(self, candidate: Path) -> tuple[str, list[dict[str, Any]]]:
        artifact_id, files = self.inventory(candidate)
        destination = self.path(artifact_id)
        if destination.exists():
            if self.inventory(destination)[0] != artifact_id:
                raise ModelPortError("INTEGRITY_ERROR", "Stored artifact digest changed")
        else:
            os.replace(candidate, destination)
        return artifact_id, files

    def verify(self, artifact_id: str) -> Path:
        path = self.path(artifact_id)
        if not path.is_dir() or self.inventory(path)[0] != artifact_id:
            raise ModelPortError("INTEGRITY_ERROR", "Artifact bytes no longer match their identity")
        return path
