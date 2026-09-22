"""Audit installed dependencies plus PyTorch's upstream version (+cpu is absent on PyPI)."""

import importlib.metadata
import json
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
output = root / ".tools"
output.mkdir(exist_ok=True)
commands = [
    [
        sys.executable,
        "-m",
        "pip_audit",
        "--local",
        "--format",
        "json",
        "--output",
        str(output / "pip-audit.json"),
    ]
]
with tempfile.TemporaryDirectory() as directory:
    requirements = Path(directory) / "torch.txt"
    upstream = importlib.metadata.version("torch").split("+")[0]
    requirements.write_text(f"torch=={upstream}\n", encoding="utf-8")
    commands.append(
        [
            sys.executable,
            "-m",
            "pip_audit",
            "-r",
            str(requirements),
            "--no-deps",
            "--disable-pip",
            "--format",
            "json",
            "--output",
            str(output / "torch-audit.json"),
        ]
    )
    statuses = [subprocess.run(command, timeout=180).returncode for command in commands]
    print(
        json.dumps(
            {
                "installed_scan_exit": statuses[0],
                "upstream_torch_scan_exit": statuses[1],
                "torch_upstream": upstream,
                "reports": str(output),
            }
        )
    )
    raise SystemExit(max(statuses))
