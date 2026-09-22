import csv
import html
import io
from pathlib import Path
from typing import Any

from modelport.security import write_json


def csv_safe(value: Any) -> Any:
    return (
        "'" + value
        if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r"))
        else value
    )


def benchmark_csv(reports: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    columns = [
        "artifact_id",
        "report_id",
        "created_at",
        "validation_state",
        "batch_size",
        "threads",
        "sample_count",
        "load_ms",
        "first_inference_ms",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "throughput_items_per_second",
        "artifact_bytes",
        "sampled_peak_rss_bytes",
    ]
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for report in reports:
        flattened = {**report, **report["config"], **report["measurements"]}
        writer.writerow({key: csv_safe(flattened.get(key)) for key in columns})
    return stream.getvalue()


def benchmark_html(reports: list[dict[str, Any]]) -> str:
    rows = []
    for report in reports:
        data, config = report["measurements"], report["config"]

        def esc(value):
            return html.escape(str(value), quote=True)

        cells = [
            report["artifact_id"][:16],
            report["validation_state"],
            config["batch_size"],
            config["threads"],
            data["sample_count"],
            f"{data['mean_ms']:.4f}",
            f"{data['p95_ms']:.4f}",
            f"{data['throughput_items_per_second']:.1f}",
            report["artifact_bytes"],
        ]
        rows.append("<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in cells) + "</tr>")
    details = "".join(
        "<details><summary>Configuration and provenance · "
        + html.escape(r["report_id"])
        + "</summary><pre>"
        + html.escape(__import__("json").dumps(r, indent=2, allow_nan=False))
        + "</pre></details>"
        for r in reports
    )
    return (
        (
            '<!doctype html><html lang="en"><meta charset="utf-8"><meta '
            'name="viewport" content="width=device-width"><title>ModelPort benchmark '
            "evidence</title><style>body{font:16px "
            "system-ui;background:#f3f5f7;color:#152230;max-width:1200px;margin:48px "
            "auto;padding:24px}h1{font-size:36px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:14px;text-align:left;border-bottom:1px"
            " solid "
            "#dfe5eb}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:20px}details{margin:20px"
            " 0}p{color:#4a5b6a}</style><h1>ModelPort / measured "
            "evidence</h1><p>Actual CPU measurements. Synthetic inputs do not "
            "establish task accuracy. Compare matching hardware, batch size and "
            "thread settings. Fresh processes do not flush the OS file "
            "cache.</p><table><thead><tr><th>Artifact</th><th>Validation</th><th>Batch</th><th>Threads</th><th>Samples</th><th>Mean"
            " · ms</th><th>P95 · ms</th><th>Items / "
            "s</th><th>Bytes</th></tr></thead><tbody>"
        )
        + "".join(rows)
        + "</tbody></table>"
        + details
        + "</html>"
    )


def export_reports(directory: Path, reports: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "benchmarks.json", {"schema_version": 1, "reports": reports})
    (directory / "benchmarks.csv").write_text(benchmark_csv(reports), encoding="utf-8")
    (directory / "benchmarks.html").write_text(benchmark_html(reports), encoding="utf-8")
