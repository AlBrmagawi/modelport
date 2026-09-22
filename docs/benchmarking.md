# Benchmark methodology

Each benchmark job launches a fresh native process. Reported load duration includes
runtime/framework initialization, architecture or session construction, and weight
loading. It is **not** a cold OS-cache measurement. First inference is recorded
separately. Inputs are generated before model-load timing.

Default settings: CPU, one intra-op thread, one ORT inter-op thread, sequential ORT,
batch 4, five warmup calls, thirty timed iterations, one repetition. These bounds
are configurable through `BenchmarkConfig`; the API validates the same schema.
Every input must have the requested leading batch size. A fixed-batch model needs
that exact `batch_size`; mismatches fail instead of overstating processed items.

`perf_counter_ns()` measures each synchronous CPU prediction. The timed boundary
includes input checking, PyTorch tensor copies if needed, runtime execution, and
NumPy output normalization. It excludes input generation, serialization, conversion,
and process RSS sampling. Both adapters use the documented high-level interface;
this is not a bare-kernel microbenchmark.

Reports preserve all millisecond samples and record population standard deviation,
min/max, p50/p95/p99, sample count, total timed duration, processed item count and
items/second from actual timed work. Quantiles with fewer than 100 observations carry
a warning. No universal speed threshold is asserted in CI.

Memory uses native process RSS sampled between calls, plus Windows OS peak working
set when available. These are distinct measures; in-call peaks may exceed sampled
RSS. GPU memory is null with a CPU-only explanation. Python allocation tracking is
not presented as total ML memory.

JSON is canonical; CSV has explicit units and formula-safe string fields. HTML is
self-contained and escapes untrusted metadata. Historical measurements retain their
timestamps and configuration. The dashboard exposes batch, thread, sample, runtime,
validation and environment differences. Compare matched settings sequentially.
