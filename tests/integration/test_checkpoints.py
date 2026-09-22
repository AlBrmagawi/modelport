import pytest
from sqlalchemy import select

from modelport.errors import ModelPortError
from modelport.persistence import checkpoints
from modelport.processes import ChildProcess


def test_retry_reuses_verified_real_export_checkpoint(workflow, monkeypatch):
    import modelport.supervisor as module

    client = workflow["client"]
    plan = client.plan(workflow["source"].artifact_id, optimize=True)
    original = ChildProcess

    def fail_second_step(args, work, memory_bytes, threads, log_limit):
        code = (
            "import modelport.native as n; from modelport.errors import ModelPortError; "
            "from modelport.runner import main; "
            "n.optimize_onnx=lambda *a: (_ for _ in ()).throw("
            "ModelPortError('CONTROLLED_FAILURE','second step interrupted')); main()"
        )
        return original(["-c", code], work, memory_bytes, threads, log_limit)

    monkeypatch.setattr(module, "ChildProcess", fail_second_step)
    job = client.convert(plan, wait=False)
    with pytest.raises(ModelPortError, match="second step interrupted"):
        client.wait(job)
    with client.repository.engine.connect() as connection:
        saved = (
            connection.execute(select(checkpoints).where(checkpoints.c.job_id == job.job_id))
            .mappings()
            .all()
        )
    assert len(saved) == 1 and saved[0]["step_index"] == 0
    assert saved[0]["data"]["step"]["evidence"]["exporter"] == "torch.export/dynamo"
    monkeypatch.setattr(module, "ChildProcess", original)
    done = client.wait(client.repository.retry(job.job_id))
    artifact = client.repository.artifact(done.result["artifact_id"])
    assert artifact.validation_state == "validated" and done.attempt == 2
    assert artifact.manifest["steps"][0]["reused_checkpoint"] == saved[0]["digest"]
    assert any(
        event["payload"].get("stage") == "checkpoint-reused"
        for event in client.repository.job_events(job.job_id)
    )
