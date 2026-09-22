"""Prepare a read-only-input diagnostic benchmark for the optional container runner."""

import argparse
from pathlib import Path

from modelport.domain import BenchmarkConfig
from modelport.sdk import ModelPort
from modelport.security import write_json

parser = argparse.ArgumentParser()
parser.add_argument("artifact_id")
parser.add_argument("--data-dir", default=".modelport")
parser.add_argument("--output", type=Path, default=Path("container-job"))
args = parser.parse_args()
with ModelPort.local(args.data_dir) as client:
    source = client.supervisor.artifact_input(args.artifact_id)
    source["path"] = "/inputs/" + args.artifact_id
    settings = client.settings.model_dump(mode="json")
    settings["data_dir"] = "/work"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results").mkdir(exist_ok=True)
    write_json(
        args.output / "request.json",
        {
            "kind": "benchmark",
            "settings": settings,
            "timeout_seconds": 180,
            "payload": {"source": source, "config": BenchmarkConfig().model_dump()},
        },
    )
print("Prepared diagnostic benchmark. See docs/deployment.md for container UID and mount setup.")
