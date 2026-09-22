import shutil
import time

from sqlalchemy import select

from modelport.domain import TERMINAL
from modelport.errors import ModelPortError
from modelport.persistence import artifacts, checkpoints, datasets, jobs, uploads
from modelport.processes import FileLock


def garbage_collect(client, *, apply: bool = False, older_than_hours: int = 24):
    if older_than_hours < 1:
        raise ModelPortError("INVALID_CONFIG", "Retention must be at least one hour")
    lock = FileLock(client.settings.data_dir / "worker.lock")
    if not lock.acquire():
        raise ModelPortError(
            "WORKER_BUSY", "Garbage collection requires an idle worker", status=409
        )
    try:
        with client.repository.transaction() as connection:
            referenced = set(connection.scalars(select(artifacts.c.artifact_id)))
            checkpoint_refs = set(connection.scalars(select(checkpoints.c.digest)))
            dataset_refs = {item["sha256"] for item in connection.scalars(select(datasets.c.data))}
            requests = list(connection.scalars(select(jobs.c.request)))
            upload_refs = set(connection.scalars(select(uploads.c.staging_id)))
            active = list(
                connection.scalars(select(jobs.c.state).where(jobs.c.state.not_in(list(TERMINAL))))
            )
            if active:
                raise ModelPortError(
                    "WORKER_BUSY", "Wait for pending jobs before garbage collection", status=409
                )
            staging_refs = upload_refs | {
                request[k]
                for request in requests
                for k in ("staging_id", "dataset_id")
                if k in request
            }
            candidates = []
            cutoff = time.time() - older_than_hours * 3600
            for name, refs in (
                ("blobs", referenced),
                ("staging", staging_refs),
                ("checkpoints", checkpoint_refs),
                ("datasets", dataset_refs),
            ):
                root = (client.settings.data_dir / name).resolve()
                for path in root.iterdir():
                    if (
                        path.name not in refs
                        and path.stat().st_mtime < cutoff
                        and path.resolve().parent == root
                        and path.is_dir()
                        and not path.is_symlink()
                    ):
                        candidates.append(
                            {
                                "area": name,
                                "id": path.name,
                                "bytes": sum(
                                    p.stat().st_size for p in path.rglob("*") if p.is_file()
                                ),
                            }
                        )
                        if apply:
                            shutil.rmtree(path)
            return {
                "dry_run": not apply,
                "candidates": candidates,
                "retention_hours": older_than_hours,
                "note": (
                    "Referenced input staging is retained for retry; records and referenced "
                    "artifacts are never removed"
                ),
            }
    finally:
        lock.close()
