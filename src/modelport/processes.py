"""Process lifetime controls. Fault isolation, not a security sandbox."""

import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, BinaryIO

import psutil

from modelport.errors import ModelPortError
from modelport.security import child_environment


class FileLock:
    def __init__(self, path: Path):
        self.path = path
        self.stream: BinaryIO | None = None

    def acquire(self, blocking: bool = False) -> bool:
        self.stream = self.path.open("a+b")
        self.stream.seek(0)
        self.stream.write(b"0")
        self.stream.flush()
        self.stream.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(
                    self.stream.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1
                )
            else:
                import fcntl

                fcntl.flock(
                    self.stream.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
                )
            return True
        except OSError:
            self.stream.close()
            self.stream = None
            return False

    def close(self) -> None:
        if self.stream:
            self.stream.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None


class WindowsJob:
    kernel: Any
    handle: Any

    def __init__(self, memory_bytes: int):
        if sys.platform != "win32":
            raise RuntimeError("Windows Job Objects require Windows")
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_uint64)
                for name in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimits),
                ("IoInfo", IO),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.kernel.CreateJobObjectW(None, None)
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = (
            0x2000 | 0x100 | 0x200
        )  # kill on close + process/job memory
        limits.ProcessMemoryLimit = memory_bytes
        limits.JobMemoryLimit = memory_bytes
        if not self.handle or not self.kernel.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            self.close()
            raise ModelPortError(
                "ISOLATION_UNAVAILABLE", "Windows Job Object resource policy could not be applied"
            )

    def attach_and_resume(self, process: subprocess.Popen) -> None:
        if sys.platform != "win32":
            raise RuntimeError("Windows Job Objects require Windows")
        import ctypes

        handle = int(process.__dict__["_handle"])
        if not self.kernel.AssignProcessToJobObject(self.handle, handle):
            process.kill()
            process.wait()
            self.close()
            raise ModelPortError(
                "ISOLATION_UNAVAILABLE", "Could not attach suspended worker to Windows Job Object"
            )
        resume = ctypes.WinDLL("ntdll").NtResumeProcess
        resume.argtypes = [ctypes.c_void_p]
        resume.restype = ctypes.c_long
        if resume(handle) != 0:
            self.close()
            raise ModelPortError("ISOLATION_UNAVAILABLE", "Could not resume supervised worker")

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class ChildProcess:
    def __init__(
        self,
        args: list[str],
        work: Path,
        memory_bytes: int,
        threads: int = 1,
        log_limit: int = 256 * 1024,
    ):
        self.job: Any = WindowsJob(memory_bytes) if sys.platform == "win32" else None
        self.output = bytearray()
        self.truncated = False
        self.limit = log_limit
        flags = 0x08000000 | 0x4 if sys.platform == "win32" else 0  # no window + suspended
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-X", "utf8", *args],
                cwd=work,
                env=child_environment(work, threads),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=flags,
                start_new_session=sys.platform != "win32",
            )
            if self.job:
                self.job.attach_and_resume(self.process)
        except BaseException:
            if self.job:
                self.job.close()
            raise
        self.reader = threading.Thread(target=self._drain, daemon=True)
        self.reader.start()

    def _drain(self) -> None:
        assert self.process.stdout is not None
        while chunk := self.process.stdout.read(4096):
            remaining = self.limit - len(self.output)
            if remaining > 0:
                self.output.extend(chunk[:remaining])
            if len(chunk) > remaining:
                self.truncated = True

    def kill(self) -> None:
        if self.job:
            self.job.close()
        elif sys.platform != "win32":
            children = []
            try:
                parent = psutil.Process(self.process.pid)
                children = parent.children(recursive=True)
            except psutil.NoSuchProcess:
                pass
            # The session/process group remains addressable if its leader has exited.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(children, timeout=5)
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.reader.join(timeout=2)

    def close(self) -> None:
        self.kill()
        if self.process.stdout:
            self.process.stdout.close()


def bind_parent_lifetime() -> None:
    if sys.platform == "linux" and (expected := os.environ.get("MODELPORT_PARENT_PID")):
        import ctypes

        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL) != 0:
            raise ModelPortError("ISOLATION_UNAVAILABLE", "Could not bind worker lifetime")
        if os.getppid() != int(expected):
            # Close the race where the parent exited before prctl was applied.
            os.kill(os.getpid(), signal.SIGKILL)


def apply_child_limits(memory_bytes: int, timeout: int) -> None:
    bind_parent_lifetime()
    if sys.platform != "win32":
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (timeout + 2, timeout + 3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024**2, 256 * 1024**2))
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
