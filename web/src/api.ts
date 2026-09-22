import type { components } from "./schema";

export type Artifact = Omit<
  components["schemas"]["ArtifactBundle"],
  "descriptor"
> & { descriptor: Required<components["schemas"]["ModelDescriptor"]> };
export type Job = components["schemas"]["JobRecord"];
export type Dataset = components["schemas"]["TensorDataset"];
export type Capabilities = components["schemas"]["CapabilitySnapshot"];
export type Plan = Omit<
  components["schemas"]["ConversionPlan"],
  "preflight"
> & { preflight: Required<components["schemas"]["PreflightReport"]> };
export type Validation = {
  report_id: string;
  source_id: string;
  target_id: string;
  state: string;
  created_at: string;
  policy: {
    name: string;
    version: string;
    gate: string;
    atol: number;
    rtol: number;
  };
  dataset: {
    kind: string;
    sample_count: number;
    batch_count: number;
    sha256: string;
  };
  outputs: {
    output: string;
    batch_index: number;
    passed: boolean;
    max_absolute_error?: number;
    normalized_l2?: number;
    cosine_similarity?: number;
    top1_agreement?: number;
    reasons: string[];
  }[];
  failures: string[];
  evidence_scope: string;
};
export type Benchmark = {
  report_id: string;
  artifact_id: string;
  created_at: string;
  validation_state: string;
  artifact_bytes: number;
  config: {
    batch_size: number;
    threads: number;
    iterations: number;
    warmup: number;
  };
  measurements: {
    mean_ms: number;
    p50_ms: number;
    p95_ms: number;
    p99_ms: number;
    load_ms: number;
    first_inference_ms: number;
    sample_count: number;
    throughput_items_per_second: number;
    sampled_peak_rss_bytes: number;
    timing_boundary: string;
  };
  environment: { os: string; processor: string };
  runtime: { runtime: string; version: string };
  warnings: string[];
};

export async function api<T>(
  token: string,
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(`/api/v1${path}`, {
    method: body === undefined ? "GET" : "POST",
    signal,
    headers: {
      Authorization: `Bearer ${token}`,
      ...(body instanceof FormData
        ? {}
        : { "Content-Type": "application/json" }),
    },
    body:
      body === undefined
        ? undefined
        : body instanceof FormData
          ? body
          : JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      `${data.code ?? response.status}: ${data.detail ?? "Request failed"}`,
    );
  return data as T;
}

export const terminal = (state: string) =>
  ["succeeded", "failed", "cancelled", "timed_out", "interrupted"].includes(
    state,
  );
export const bytes = (value: number) =>
  value >= 1024 * 1024
    ? `${(value / 1024 / 1024).toFixed(2)} MiB`
    : `${(value / 1024).toFixed(1)} KiB`;
export const number = (value: number | null | undefined, digits = 4) =>
  value == null
    ? "—"
    : value.toLocaleString(undefined, { maximumFractionDigits: digits });
export const short = (value: string) => value.slice(0, 10);

export function parseEvent(
  block: string,
): { sequence: number; kind: string; payload: Record<string, unknown> } | null {
  const line = block.split("\n").find((line) => line.startsWith("data: "));
  return line ? JSON.parse(line.slice(6)) : null;
}

export async function watchJob(
  token: string,
  id: string,
  onEvent: (event: NonNullable<ReturnType<typeof parseEvent>>) => void,
  signal: AbortSignal,
) {
  let cursor = 0;
  while (!signal.aborted) {
    const response = await fetch(`/api/v1/jobs/${id}/events`, {
      headers: {
        Authorization: `Bearer ${token}`,
        "Last-Event-ID": String(cursor),
      },
      signal,
    });
    if (!response.ok || !response.body)
      throw new Error(`Job event connection failed (${response.status})`);
    const reader = response.body.getReader(),
      decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary: number;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const item = parseEvent(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
        if (item && item.sequence > cursor) {
          cursor = item.sequence;
          onEvent(item);
          if (item.kind === "state" && terminal(String(item.payload.state))) {
            await reader.cancel();
            return;
          }
        }
      }
    }
    const job = await api<Job>(token, `/jobs/${id}`, undefined, signal);
    if (terminal(job.state)) return;
    await new Promise((resolve) => setTimeout(resolve, 600));
  }
}

export async function download(token: string, path: string, filename: string) {
  const response = await fetch(`/api/v1${path}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) throw new Error(`Download failed (${response.status})`);
  const url = URL.createObjectURL(await response.blob()),
    link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
