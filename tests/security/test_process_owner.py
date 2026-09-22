import os
import subprocess
import sys
import time

import psutil
import pytest


@pytest.mark.skipif(
    os.name != "nt", reason="Windows Job Object owner-death semantics are OS-specific"
)
def test_forced_supervisor_exit_terminates_native_descendants(tmp_path):
    code = """
import pathlib,time
from modelport.processes import ChildProcess
work=pathlib.Path.cwd()
child=ChildProcess(['-c', 'import time; time.sleep(120)'],work,1024**3)
(work/'child.pid').write_text(str(child.process.pid))
time.sleep(120)
"""
    parent = subprocess.Popen([sys.executable, "-c", code], cwd=tmp_path, creationflags=0x08000000)
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "child.pid").exists() and time.monotonic() < deadline:
            time.sleep(0.03)
        child_pid = int((tmp_path / "child.pid").read_text())
        assert psutil.pid_exists(child_pid)
        parent.kill()
        parent.wait(timeout=5)
        deadline = time.monotonic() + 5
        while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
            time.sleep(0.03)
        assert not psutil.pid_exists(child_pid)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux parent-death signal semantics")
def test_forced_linux_owner_exit_terminates_bound_worker_and_runner(tmp_path):
    runner = (
        "from modelport.processes import bind_parent_lifetime; "
        "import time; bind_parent_lifetime(); time.sleep(120)"
    )
    worker = f"""
import pathlib,time
from modelport.processes import ChildProcess, bind_parent_lifetime
bind_parent_lifetime()
work=pathlib.Path.cwd()
child=ChildProcess(['-c', {runner!r}],work,1024**3)
(work/'runner.pid').write_text(str(child.process.pid))
time.sleep(120)
"""
    owner = f"""
import pathlib,time
from modelport.processes import ChildProcess
work=pathlib.Path.cwd()
child=ChildProcess(['-c', {worker!r}],work,1024**3)
(work/'worker.pid').write_text(str(child.process.pid))
time.sleep(120)
"""
    parent = subprocess.Popen([sys.executable, "-c", owner], cwd=tmp_path)
    pids = []
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "runner.pid").exists() and time.monotonic() < deadline:
            time.sleep(0.03)
        pids = [int((tmp_path / f"{name}.pid").read_text()) for name in ("worker", "runner")]
        parent.kill()
        parent.wait(timeout=5)

        def alive(pid):
            try:
                return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                return False

        deadline = time.monotonic() + 5
        while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.03)
        assert not any(alive(pid) for pid in pids)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        for pid in pids:
            try:
                psutil.Process(pid).kill()
            except psutil.NoSuchProcess:
                pass
