"""Conservative cross-platform file and tensor boundaries; no archive extraction."""

import hashlib
import json
import math
import os
import re
import stat
import struct
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from modelport.errors import ModelPortError

RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(10)),
    *(f"lpt{i}" for i in range(10)),
}


def canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def identity(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(name: str) -> str:
    if not name or len(name) > 240 or "\\" in name or ":" in name or "\x00" in name:
        raise ModelPortError(
            "UNSAFE_PATH", "Use a short, relative POSIX filename without drive or stream names"
        )
    path = PurePosixPath(name)
    if path.is_absolute() or PureWindowsPath(name).drive:
        raise ModelPortError("UNSAFE_PATH", "Absolute paths are not accepted")
    for part in name.split("/"):
        if (
            part in {"", ".", ".."}
            or part.endswith((" ", "."))
            or part.split(".")[0].lower() in RESERVED
            or not re.fullmatch(r"[A-Za-z0-9_ .-]+", part)
        ):
            raise ModelPortError(
                "UNSAFE_PATH", "Filename contains traversal, reserved or unsupported characters"
            )
    return name


def reject_link(path: Path) -> None:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ModelPortError(
            "UNSAFE_PATH", "Symbolic links, junctions and reparse points are rejected"
        )


def confined(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    safe_relative(relative)
    root = root.resolve()
    path = root / relative
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.exists():
            reject_link(current)
    if not path.resolve().is_relative_to(root):
        raise ModelPortError("UNSAFE_PATH", "Path escapes its owning bundle")
    if must_exist and not path.is_file():
        raise ModelPortError("MISSING_COMPANION", "Referenced companion file is missing")
    return path


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ModelPortError("INVALID_METADATA", "Duplicate JSON keys are ambiguous")
        result[key] = value
    return result


def read_json(path: Path, limit: int = 1024**2) -> Any:
    if path.stat().st_size > limit:
        raise ModelPortError("RESOURCE_EXHAUSTED", "JSON metadata exceeds the configured limit")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON value")),
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ModelPortError(
            "INVALID_METADATA", "Malformed, nonfinite or deeply nested JSON"
        ) from exc


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(canonical(value), encoding="utf-8")
    os.replace(temporary, path)


def safetensors_header(path: Path, max_bytes: int, max_metadata: int) -> dict[str, Any]:
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ModelPortError("INVALID_MODEL", "Truncated SafeTensors header")
        length = struct.unpack("<Q", length_bytes)[0]
        if length > max_metadata or length < 2:
            raise ModelPortError("RESOURCE_EXHAUSTED", "SafeTensors header exceeds metadata budget")
        raw = stream.read(length)
    try:
        header = json.loads(raw, object_pairs_hook=unique_pairs)
        if not isinstance(header, dict) or len(header) > 10_000:
            raise ValueError("Invalid tensor map")
        tensors, total, ranges = [], 0, []
        sizes = {
            "F32": 4,
            "F64": 8,
            "F16": 2,
            "BF16": 2,
            "I64": 8,
            "I32": 4,
            "I16": 2,
            "I8": 1,
            "U8": 1,
            "BOOL": 1,
        }
        for name, item in header.items():
            if name == "__metadata__":
                if not isinstance(item, dict) or any(not isinstance(v, str) for v in item.values()):
                    raise ValueError("Invalid metadata")
                continue
            shape, dtype, offsets = item["shape"], item["dtype"], item["data_offsets"]
            if len(name) > 256 or len(shape) > 8 or any(type(d) is not int or d < 0 for d in shape):
                raise ValueError("Invalid shape or name")
            size = math.prod(shape) * sizes[dtype]
            start, end = offsets
            if type(start) is not int or type(end) is not int or start < 0 or end - start != size:
                raise ValueError("Invalid offsets")
            if 8 + length + end > path.stat().st_size:
                raise ValueError("Truncated tensor bytes")
            ranges.append((start, end))
            total += size
            tensors.append({"name": name, "shape": shape, "dtype": dtype, "size_bytes": size})
        if total > max_bytes:
            raise ModelPortError("RESOURCE_EXHAUSTED", "Tensor allocation budget exceeded")
        end = 0
        for start, stop in sorted(ranges):
            if start != end:
                raise ValueError("Overlapping tensors or holes in data")
            end = stop
        if 8 + length + end != path.stat().st_size:
            raise ValueError("Unexpected trailing bytes")
        return {
            "tensors": tensors,
            "metadata": header.get("__metadata__", {}),
            "tensor_bytes": total,
        }
    except (KeyError, TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ModelPortError("INVALID_MODEL", "Malformed or truncated SafeTensors data") from exc


def load_npz(path: Path, limit: int = 128 * 1024**2) -> dict[str, Any]:
    import numpy as np

    if path.stat().st_size > limit:
        raise ModelPortError("RESOURCE_EXHAUSTED", "Input dataset exceeds byte limit")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > 16 or sum(x.file_size for x in entries) > limit:
                raise ModelPortError("RESOURCE_EXHAUSTED", "Expanded dataset exceeds limit")
            seen = set()
            for entry in entries:
                safe_relative(entry.filename)
                if (
                    "/" in entry.filename
                    or entry.filename in seen
                    or not entry.filename.endswith(".npy")
                ):
                    raise ModelPortError(
                        "INVALID_INPUT", "NPZ must contain unique named NPY tensors"
                    )
                seen.add(entry.filename)
                # Check NPY header before NumPy allocates a claimed enormous array.
                with archive.open(entry) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version not in {(1, 0), (2, 0)}:
                        raise ModelPortError("INVALID_INPUT", "Unsupported NPY header version")
                    reader = (
                        np.lib.format.read_array_header_1_0
                        if version == (1, 0)
                        else np.lib.format.read_array_header_2_0
                    )
                    shape, _, dtype = reader(stream, max_header_size=8192)
                    if (
                        dtype.hasobject
                        or len(shape) > 8
                        or math.prod(shape) * dtype.itemsize > limit
                    ):
                        raise ModelPortError(
                            "INVALID_INPUT", "Object tensors or excessive shapes rejected"
                        )
                    if math.prod(shape) * dtype.itemsize > entry.file_size - stream.tell():
                        raise ModelPortError("INVALID_INPUT", "Truncated NPY tensor")
        with np.load(path, allow_pickle=False, max_header_size=8192) as data:
            return {key: data[key].copy() for key in data.files}
    except (ValueError, OSError, zipfile.BadZipFile, EOFError) as exc:
        raise ModelPortError(
            "INVALID_INPUT", "Invalid NPZ dataset; pickle is never enabled"
        ) from exc


def save_npz(path: Path, arrays: dict[str, Any]) -> None:
    """Write tensor names literally, including NumPy's reserved argument names."""
    import numpy as np

    if not arrays or len(arrays) > 16:
        raise ModelPortError("INVALID_INPUT", "Expected one to sixteen named tensors")
    for name, array in arrays.items():
        safe_relative(name)
        if "/" in name or not isinstance(array, np.ndarray) or array.dtype.hasobject:
            raise ModelPortError("INVALID_INPUT", "Expected flat tensor names and numeric arrays")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in arrays.items():
            with archive.open(name + ".npy", "w") as stream:
                np.lib.format.write_array(stream, array, allow_pickle=False)


def child_environment(work: Path, threads: int = 1) -> dict[str, str]:
    allowed = ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATH", "PATHEXT", "LD_LIBRARY_PATH")
    env = {name: os.environ[name] for name in allowed if name in os.environ}
    env.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "TEMP": str(work),
            "TMP": str(work),
            "TMPDIR": str(work),
            "HOME": str(work),
            "USERPROFILE": str(work),
            "USERNAME": "modelport-worker",
            "USER": "modelport-worker",
            "TORCHINDUCTOR_CACHE_DIR": str(work / "torch-cache"),
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "MODELPORT_PARENT_PID": str(os.getpid()),
        }
    )
    return env
