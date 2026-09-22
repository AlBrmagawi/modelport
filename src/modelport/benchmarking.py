import os
import platform
import time
import uuid
from pathlib import Path
from typing import Any

from modelport.config import Settings
from modelport.domain import BenchmarkConfig, BenchmarkReport, ModelDescriptor
from modelport.errors import ModelPortError
from modelport.tensors import synthetic


def environment() -> dict[str, Any]:
    import psutil

    return {
        "os": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "logical_cores": os.cpu_count(),
        "physical_cores": psutil.cpu_count(logical=False),
        "ram_bytes": psutil.virtual_memory().total,
        "python": platform.python_version(),
    }


def benchmark_model(
    path: Path,
    descriptor: ModelDescriptor,
    artifact_id: str,
    validation_state: str,
    settings: Settings,
    config: BenchmarkConfig,
) -> BenchmarkReport:
    import numpy as np
    import psutil

    inputs = synthetic(descriptor.inputs, config.batch_size, config.seed)
    if any(value.ndim == 0 or value.shape[0] != config.batch_size for value in inputs.values()):
        raise ModelPortError(
            "INVALID_BATCH_SIZE",
            "Benchmark inputs must share the requested leading batch dimension; "
            "choose the model's fixed batch size when its signature is static",
        )
    start = time.perf_counter_ns()
    from modelport.native import NativeModel

    runtime = NativeModel(path, descriptor, settings, config.threads)
    load_ms = (time.perf_counter_ns() - start) / 1e6
    process = psutil.Process()
    rss = [process.memory_info().rss]
    start = time.perf_counter_ns()
    runtime.predict(inputs)
    first_ms = (time.perf_counter_ns() - start) / 1e6
    try:
        for _ in range(config.warmup):
            runtime.predict(inputs)
        samples = []
        for _ in range(config.repetitions):
            for _ in range(config.iterations):
                start = time.perf_counter_ns()
                runtime.predict(inputs)
                samples.append((time.perf_counter_ns() - start) / 1e6)
                rss.append(process.memory_info().rss)
        memory = process.memory_info()
        measured = float(sum(samples))
        measurements = {
            "load_ms": load_ms,
            "first_inference_ms": first_ms,
            "latency_samples_ms": samples,
            "sample_count": len(samples),
            "mean_ms": float(np.mean(samples)),
            "std_ms": float(np.std(samples)),
            "min_ms": min(samples),
            "max_ms": max(samples),
            **{f"p{p}_ms": float(np.percentile(samples, p)) for p in (50, 95, 99)},
            "total_timed_ms": measured,
            "processed_items": len(samples) * config.batch_size,
            "throughput_items_per_second": len(samples) * config.batch_size / (measured / 1000),
            "sampled_peak_rss_bytes": max(rss),
            "os_peak_rss_bytes": getattr(memory, "peak_wset", None),
            "os_peak_rss_note": None
            if hasattr(memory, "peak_wset")
            else "Not collected on this OS",
            "gpu_memory_bytes": None,
            "gpu_memory_note": "CPU runtime",
            "input_shapes": {n: list(v.shape) for n, v in inputs.items()},
            "timing_boundary": (
                "input validation + tensor conversion/copy + model execution + NumPy "
                "output; excludes generation, serialization and RSS sampling"
            ),
            "load_boundary": (
                "fresh worker process; native imports/session or module construction + "
                "weights; OS page cache is not flushed"
            ),
        }
        return BenchmarkReport(
            report_id=uuid.uuid4().hex,
            artifact_id=artifact_id,
            config=config,
            measurements=measurements,
            environment=environment(),
            runtime=runtime.info,
            validation_state=validation_state,
            artifact_bytes=sum(p.stat().st_size for p in path.rglob("*") if p.is_file()),
            warnings=["Tail percentiles have fewer than 100 observations"]
            if len(samples) < 100
            else [],
        )
    finally:
        runtime.close()
