"""Persistent, single-caller native session for the local SDK."""

import time
from pathlib import Path

from modelport.config import Settings
from modelport.domain import ModelDescriptor
from modelport.errors import ModelPortError
from modelport.processes import apply_child_limits
from modelport.security import load_npz, read_json, save_npz, write_json


def main():
    work = Path.cwd()
    config = read_json(work / "session.json", 4 * 1024**2)
    settings = Settings.model_validate(config["settings"])
    apply_child_limits(settings.memory_limit_bytes, 3600)
    model = None
    try:
        from modelport.native import NativeModel

        model = NativeModel(
            Path(config["path"]), ModelDescriptor.model_validate(config["descriptor"]), settings
        )
        write_json(work / "ready.json", {"ok": True, "runtime": model.info})
        index = 0
        while True:
            request = work / f"request-{index}.json"
            if not request.exists():
                time.sleep(0.02)
                continue
            read_json(request, 1024)
            try:
                outputs = model.predict(load_npz(work / f"input-{index}.npz"))
                if sum(value.nbytes for value in outputs.values()) > settings.max_tensor_bytes:
                    raise ModelPortError("RESOURCE_EXHAUSTED", "Output tensor budget exceeded")
                save_npz(work / f"output-{index}.npz", outputs)
                write_json(work / f"result-{index}.json", {"ok": True})
            except ModelPortError as exc:
                write_json(work / f"result-{index}.json", {"ok": False, "error": exc.problem()})
            index += 1
    except Exception as exc:
        write_json(
            work / "ready.json",
            {
                "ok": False,
                "error": {
                    "code": getattr(exc, "code", "RUNTIME_FAILED"),
                    "detail": getattr(
                        exc, "detail", "Native session failed; check artifact compatibility"
                    ),
                },
            },
        )
    finally:
        if model:
            model.close()


if __name__ == "__main__":
    main()
