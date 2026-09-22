"""Fail-fast local Python release checks; no external services required."""

import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
commands = [
    ["-m", "ruff", "check", "src", "tests", "scripts", "examples", "hatch_build.py"],
    ["-m", "ruff", "format", "--check", "src", "tests", "scripts", "examples", "hatch_build.py"],
    ["-m", "mypy", "src/modelport"],
    ["scripts/schema.py", "--check"],
    ["-m", "pytest", "-q", "--junitxml=.tools/pytest.xml"],
]
for arguments in commands:
    completed = subprocess.run([sys.executable, *arguments], cwd=root, timeout=600)
    if completed.returncode:
        raise SystemExit(completed.returncode)
