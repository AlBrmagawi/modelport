import csv
import os
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("MODELPORT_DATA_DIR", ".modelport"))
    )
    max_file_bytes: int = 128 * 1024**2
    max_bundle_bytes: int = 256 * 1024**2
    max_files: int = 64
    max_metadata_bytes: int = 1024**2
    max_tensor_bytes: int = 128 * 1024**2
    max_log_bytes: int = 256 * 1024
    max_ipc_bytes: int = 4 * 1024**2
    timeout_seconds: int = 180
    memory_limit_bytes: int = 4 * 1024**3
    max_queued_jobs: int = 64
    threads: int = 1

    def prepare(self) -> None:
        self.data_dir = self.data_dir.resolve()
        if (
            self.data_dir.is_dir()
            and next(self.data_dir.iterdir(), None) is not None
            and not (self.data_dir / "modelport.db").is_file()
            and not (self.data_dir / ".private-acl").is_file()
        ):
            from modelport.errors import ModelPortError

            raise ModelPortError(
                "INVALID_CONFIG", "Choose an empty or existing ModelPort data directory"
            )
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "nt":
            # Protect only the explicitly configured application data root.
            # Existing model source paths and the project workspace are untouched.
            marker = self.data_dir / ".private-acl"
            if not marker.exists():
                from modelport.errors import ModelPortError

                user = subprocess.run(
                    ["whoami", "/user", "/fo", "csv", "/nh"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                    creationflags=0x08000000,
                )
                sid = next(csv.reader(user.stdout.splitlines()))[1]
                secured = subprocess.run(
                    [
                        "icacls",
                        str(self.data_dir),
                        "/inheritance:r",
                        "/grant:r",
                        f"*{sid}:(OI)(CI)F",
                        "/grant:r",
                        "*S-1-5-18:(OI)(CI)F",
                    ],
                    capture_output=True,
                    timeout=10,
                    creationflags=0x08000000,
                )
                if secured.returncode:
                    raise ModelPortError(
                        "PERMISSIONS_FAILED",
                        "Could not restrict the application data directory ACL",
                    )
                marker.touch()
        for name in ("blobs", "staging", "work", "uploads", "checkpoints", "datasets"):
            (self.data_dir / name).mkdir(exist_ok=True, mode=0o700)
        if os.name != "nt":
            self.data_dir.chmod(0o700)
