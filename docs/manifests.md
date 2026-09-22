# Artifact identity, publication and provenance

File identity is SHA-256, computed incrementally. Bundle identity is SHA-256 of a
canonical JSON list sorted by safe relative path, each entry containing path, size
and file digest. Timestamps are absent from this inventory. External tensor data
files participate in identity. Identical source bytes deduplicate to the same bundle.

Conversion manifests contain source inventory, root reference, sanitized origin,
architecture/code/environment identity, ordered adapter versions, requested and
effective options, exporter/opset, signatures, precision details, time/duration,
warnings, fallback policy and output inventory. The manifest's output inventory
excludes the manifest itself. The complete bundle hash includes the manifest. This
avoids self-referential hashes. Different conversion execution provenance may give
different bundle IDs even if the model graph bytes match.

Later validation and benchmark reports are immutable rows referring to bundle IDs.
They do not mutate a conversion manifest. The database's current validation summary
is mutable derived state; the underlying reports remain independently inspectable.
Downloads include `modelport-provenance.json` and `modelport-evidence.json` in addition
to the immutable files. Exported ZIP archives are not accepted for import.

The child writes into a private work directory and emits a bounded result after
structural and runtime checks. The supervisor rehashes it, starts a short transaction,
checks the current fencing token and cancellation intent, publishes by directory
rename, and inserts metadata/evidence. Filesystem and SQLite are not one atomic
transaction. A crash after rename but before commit leaves an unreferenced blob;
it is invisible through artifact APIs and eligible for dry-run garbage collection.

Checkpoint outputs are separate from usable artifacts, stored in `checkpoints/`.
Complete native steps are rehashed and bound to the exact plan ID. Retry uses a new
attempt, preserves earlier history, and skips only compatible complete steps. The
interrupted step itself restarts; partial files never become a checkpoint or artifact.
There is no cross-job conversion cache. Benchmarks always execute afresh.

A checksum establishes integrity relative to an observed or supplied value. It does
not authenticate an untrusted publisher or establish model safety.
