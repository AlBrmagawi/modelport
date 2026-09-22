import json
import subprocess
import sys


def invoke(*args, timeout=30):
    return subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "modelport.cli", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_help_is_lazy_and_lists_only_implemented_commands():
    code = (
        "import sys; from typer.testing import CliRunner; from modelport.cli import app; "
        "r=CliRunner().invoke(app,['--help']); assert r.exit_code == 0; "
        "assert 'torch' not in sys.modules; assert 'onnxruntime' not in sys.modules; print('lazy')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "lazy"


def test_json_stdout_and_stable_error_exit(tmp_path):
    result = invoke("--data-dir", str(tmp_path), "jobs", "list")
    assert result.returncode == 0
    assert json.loads(result.stdout) == [] and "\x1b" not in result.stdout
    result = invoke("--data-dir", str(tmp_path), "inspect", "missing")
    assert result.returncode == 2
    assert json.loads(result.stdout)["code"] == "NOT_FOUND"
