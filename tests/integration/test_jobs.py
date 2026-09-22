import threading
import time

import psutil
import pytest
from sqlalchemy import select, update

from modelport.errors import ModelPortError
from modelport.persistence import attempts, jobs
from modelport.processes import ChildProcess
from modelport.supervisor import Supervisor


def test_atomic_claim_and_stale_attempt_fencing(client):
    queued = client.repository.submit("fixture", {})
    claims = []
    threads = [
        threading.Thread(target=lambda: claims.append(client.repository.claim())) for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(claim is not None for claim in claims) == 1
    claim = next(claim for claim in claims if claim)
    with client.repository.transaction() as connection:
        assert not client.repository.owns(connection, queued.job_id, "stale-token")
        connection.execute(update(jobs).where(jobs.c.job_id == queued.job_id).values(heartbeat=0))
    assert client.repository.recover() == 1
    assert client.repository.job(queued.job_id).state == "interrupted"
    client.repository.retry(queued.job_id)
    new = client.repository.claim()
    with client.repository.transaction() as connection:
        assert not client.repository.owns(connection, queued.job_id, claim["token"])
        assert client.repository.owns(connection, queued.job_id, new["token"])
        assert not client.repository.finish(connection, queued.job_id, claim["token"], "succeeded")


def test_queued_cancel_idempotent_and_retry_history(client):
    job = client.repository.submit("fixture", {})
    assert client.repository.cancel(job.job_id).state == "cancelled"
    assert client.repository.cancel(job.job_id).state == "cancelled"
    assert client.repository.claim() is None
    retried = client.repository.retry(job.job_id)
    assert retried.state == "queued"
    done = client.wait(retried)
    assert done.state == "succeeded" and done.attempt == 1
    with pytest.raises(ModelPortError):
        client.repository.retry(job.job_id)


@pytest.mark.parametrize("mode", ["cancel", "timeout", "crash", "shutdown"])
def test_controlled_long_operation_kills_tree_and_never_publishes(client, monkeypatch, mode):
    # The test substitutes only the process workload. Production supervision,
    # ownership, termination and publication logic execute unchanged.
    import modelport.supervisor as module

    pids = []
    original = ChildProcess

    def controlled(args, work, memory_bytes, threads, log_limit):
        child_code = "import time; time.sleep(120)"
        if mode == "crash":
            code = "import os; os._exit(17)"
        else:
            code = (
                "import subprocess,sys,time,pathlib; "
                f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
                "pathlib.Path('descendant.txt').write_text(str(p.pid)); time.sleep(120)"
            )
        child = original(["-c", code], work, memory_bytes, threads, log_limit)
        pids.append(child.process.pid)
        return child

    monkeypatch.setattr(module, "ChildProcess", controlled)
    job = client.repository.submit("fixture", {}, timeout=1 if mode == "timeout" else 30)
    supervisor = Supervisor(client.settings, client.repository)
    thread = threading.Thread(target=supervisor.run_one)
    thread.start()
    deadline = time.monotonic() + 10
    descendant = None
    while time.monotonic() < deadline:
        paths = list((client.settings.data_dir / "work").rglob("descendant.txt"))
        if paths:
            descendant = int(paths[0].read_text())
            break
        if mode == "crash" and not thread.is_alive():
            break
        time.sleep(0.02)
    if mode == "cancel":
        assert descendant
        client.repository.cancel(job.job_id)
    if mode == "shutdown":
        supervisor.close()
    thread.join(timeout=10)
    assert not thread.is_alive()
    expected = {
        "cancel": "cancelled",
        "timeout": "timed_out",
        "crash": "failed",
        "shutdown": "interrupted",
    }[mode]
    assert client.repository.job(job.job_id).state == expected
    assert not client.repository.list_artifacts()
    assert all(not psutil.pid_exists(pid) for pid in [*pids, *([descendant] if descendant else [])])
    with client.repository.engine.connect() as connection:
        saved = (
            connection.execute(select(attempts).where(attempts.c.job_id == job.job_id))
            .mappings()
            .one()
        )
        assert saved["state"] == expected and saved["finished_at"]


def test_publication_crash_orphan_is_not_visible_and_gc_handles_it(client, monkeypatch):
    from modelport.maintenance import garbage_collect

    original = client.repository.save_artifact

    def fail(*_):
        raise RuntimeError("Simulated DB failure after filesystem publication")

    monkeypatch.setattr(client.repository, "save_artifact", fail)
    job = client.fixture(wait=False)
    with pytest.raises(ModelPortError):
        client.wait(job)
    assert not client.repository.list_artifacts()
    orphan = next((client.settings.data_dir / "blobs").iterdir())
    import os

    os.utime(orphan, (0, 0))
    result = garbage_collect(client, older_than_hours=1)
    assert result["dry_run"] and len(result["candidates"]) == 1 and orphan.exists()
    garbage_collect(client, apply=True, older_than_hours=1)
    assert not orphan.exists()
    monkeypatch.setattr(client.repository, "save_artifact", original)
    done = client.wait(client.repository.retry(job.job_id))
    assert done.state == "succeeded" and done.attempt == 2
