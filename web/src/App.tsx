import { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  ArrowLeftRight,
  ArrowRight,
  Box,
  Check,
  ChevronRight,
  CircleHelp,
  Cpu,
  Database,
  FileCode2,
  FlaskConical,
  Layers3,
  LoaderCircle,
  LogOut,
  Play,
  Plus,
  Radio,
  RefreshCw,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Upload,
  X,
} from "lucide-react";
import type { ReactNode } from "react";
import { api, bytes, download, number, short, terminal, watchJob } from "./api";
import type {
  Artifact,
  Benchmark,
  Capabilities,
  Job,
  Plan,
  Validation,
  Dataset,
} from "./api";

type Page = "library" | "jobs" | "benchmarks" | "environment";
type Tab = "inspection" | "convert" | "validation" | "provenance";
const pages: { id: Page; label: string; icon: typeof Box }[] = [
  { id: "library", label: "Model library", icon: Layers3 },
  { id: "jobs", label: "Job activity", icon: Activity },
  { id: "benchmarks", label: "Benchmarks", icon: ArrowLeftRight },
  { id: "environment", label: "Environment", icon: Cpu },
];

function Badge({ state }: { state: string }) {
  return (
    <span className={`badge ${state}`}>
      <span />
      {state.replaceAll("_", " ")}
    </span>
  );
}
function Empty({
  icon,
  title,
  children,
}: {
  icon: ReactNode;
  title: string;
  children: ReactNode;
}) {
  return (
    <div className="empty">
      {icon}
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}
function Logo() {
  return (
    <span className="logo-mark">
      <span />
      <span />
      <span />
    </span>
  );
}

export default function App() {
  const [token, setToken] = useState(""),
    [draftToken, setDraftToken] = useState("");
  const [page, setPage] = useState<Page>("library"),
    [models, setModels] = useState<Artifact[]>([]),
    [jobs, setJobs] = useState<Job[]>([]);
  const [caps, setCaps] = useState<Capabilities | null>(null),
    [selected, setSelected] = useState(""),
    [tab, setTab] = useState<Tab>("inspection");
  const [validations, setValidations] = useState<Validation[]>([]),
    [benchmarks, setBenchmarks] = useState<Benchmark[]>([]);
  const [error, setError] = useState(""),
    [busy, setBusy] = useState(""),
    [connected, setConnected] = useState(false),
    [loading, setLoading] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false),
    [files, setFiles] = useState<File[]>([]),
    [search, setSearch] = useState("");
  const [precision, setPrecision] = useState("fp32"),
    [optimize, setOptimize] = useState(false),
    [plan, setPlan] = useState<Plan | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [calibrationId, setCalibrationId] = useState("");
  const [validationId, setValidationId] = useState("");
  const [importSource, setImportSource] = useState("local");
  const [importManifest, setImportManifest] = useState("");
  const [activeJob, setActiveJob] = useState(""),
    [events, setEvents] = useState<string[]>([]),
    [logs, setLogs] = useState("");
  const [batch, setBatch] = useState(4),
    [iterations, setIterations] = useState(30);
  const mounted = useRef(true);
  const modalRef = useRef<HTMLElement | null>(null);
  useEffect(() => {
    if (!uploadOpen) return;
    const previous = document.activeElement as HTMLElement | null;
    const controls = () =>
      Array.from(
        modalRef.current?.querySelectorAll<HTMLElement>(
          "button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled)",
        ) ?? [],
      );
    controls()[0]?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) setUploadOpen(false);
      if (event.key !== "Tab") return;
      const list = controls(),
        first = list[0],
        last = list[list.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      previous?.focus();
    };
  }, [uploadOpen, busy]);
  const model = models.find((item) => item.artifact_id === selected);
  const running = jobs.filter((job) => !terminal(job.state));
  const selectedReports = validations.filter(
    (report) => report.target_id === selected,
  );

  const refresh = useCallback(
    async (credential = token) => {
      if (!credential) return;
      try {
        const [m, j, v, b, d] = await Promise.all([
          api<Artifact[]>(credential, "/models?limit=200"),
          api<Job[]>(credential, "/jobs?limit=100"),
          api<Validation[]>(credential, "/reports?kind=validation"),
          api<Benchmark[]>(credential, "/reports?kind=benchmark"),
          api<Dataset[]>(credential, "/datasets"),
        ]);
        if (!mounted.current) return;
        setModels(m);
        setJobs(j);
        setValidations(v);
        setBenchmarks(b);
        setDatasets(d);
        setConnected(true);
        try {
          setCaps(await api<Capabilities>(credential, "/capabilities"));
        } catch {
          setCaps(null);
        }
      } catch (cause) {
        setConnected(false);
        throw cause;
      }
    },
    [token],
  );

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  useEffect(() => {
    if (!token) return;
    const timer = setInterval(() => {
      void refresh().catch(() => setConnected(false));
    }, 2500);
    return () => clearInterval(timer);
  }, [token, refresh]);
  useEffect(() => {
    setPlan(null);
    setPrecision("fp32");
    setOptimize(false);
  }, [selected]);
  useEffect(() => {
    if (!activeJob || !token) return;
    setEvents([]);
    setLogs("");
    const controller = new AbortController();
    void watchJob(
      token,
      activeJob,
      (event) => {
        setEvents((previous) => [
          ...previous.slice(-30),
          event.kind === "stage"
            ? String(event.payload.stage)
            : String(event.payload.state),
        ]);
        void refresh().catch(() => {});
      },
      controller.signal,
    ).catch((cause) => {
      if (!controller.signal.aborted) setError(String(cause));
    });
    return () => controller.abort();
  }, [activeJob, token, refresh]);

  async function action(label: string, work: () => Promise<void>) {
    setError("");
    setBusy(label);
    try {
      await work();
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy("");
    }
  }
  async function waitJob(job: Job): Promise<Job> {
    setActiveJob(job.job_id);
    for (;;) {
      const current = await api<Job>(token, `/jobs/${job.job_id}`);
      if (current.state === "succeeded") return current;
      if (terminal(current.state))
        throw new Error(
          `${current.state}: ${String(current.error?.detail ?? "Job did not complete")}`,
        );
      await new Promise((resolve) => setTimeout(resolve, 400));
    }
  }
  async function login(event: React.FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      await refresh(draftToken);
      setToken(draftToken);
      setDraftToken("");
    } catch (cause) {
      setError(String(cause));
    } finally {
      setLoading(false);
    }
  }
  async function importFiles() {
    let job: Job;
    if (importSource === "local") {
      const form = new FormData();
      files.forEach((file) => form.append("files", file));
      const upload = await api<{ upload_id: string }>(token, "/uploads", form);
      job = await api<Job>(token, "/imports", { upload_id: upload.upload_id });
    } else {
      job = await api<Job>(
        token,
        `/imports/${importSource}`,
        JSON.parse(importManifest),
      );
    }
    const done = await waitJob(job);
    setSelected(String(done.result?.artifact_id));
    setTab("inspection");
    setUploadOpen(false);
    setFiles([]);
    setImportManifest("");
  }
  async function launchDemo() {
    const source = await waitJob(
      await api<Job>(token, "/fixtures", { architecture: "vision-mlp" }),
    );
    const sourceId = String(source.result?.artifact_id);
    setSelected(sourceId);
    const variants = [sourceId];
    for (const [requestedPrecision, requestedOptimize] of [
      ["fp32", false],
      ["fp32", true],
      ["dynamic-int8", false],
    ] as const) {
      setBusy(`Demo · ${requestedOptimize ? "optimize" : requestedPrecision}`);
      const route = await api<Plan>(token, "/plans", {
        source_id:
          requestedPrecision === "fp32" && !requestedOptimize
            ? sourceId
            : variants[1],
        precision: requestedPrecision,
        optimize: requestedOptimize,
      });
      const done = await waitJob(await api<Job>(token, "/conversions", route));
      variants.push(String(done.result?.artifact_id));
      await refresh();
    }
    for (const artifact_id of variants) {
      setBusy("Demo · measuring CPU inference");
      await waitJob(await api<Job>(token, "/benchmarks", { artifact_id }));
    }
    setSelected(variants[1]);
    setTab("validation");
  }
  async function importDataset(
    file: File,
    purpose: "calibration" | "validation",
  ) {
    const form = new FormData();
    form.append("file", file);
    const done = await waitJob(
      await api<Job>(token, `/datasets?purpose=${purpose}`, form),
    );
    const identifier = String(done.result?.dataset_id);
    if (purpose === "calibration") setCalibrationId(identifier);
    else setValidationId(identifier);
    setPlan(null);
    await refresh();
  }
  const canQuantize = caps?.adapters.some(
    (adapter) =>
      adapter.id === "onnx-dynamic-int8" &&
      adapter.availability === "available",
  );
  const canOptimize = caps?.adapters.some(
    (adapter) =>
      adapter.id === "onnx-optimize" && adapter.availability === "available",
  );
  const canStaticQuantize = caps?.adapters.some(
    (adapter) =>
      adapter.id === "onnx-static-int8" && adapter.availability === "available",
  );
  const canFloat16 = caps?.adapters.some(
    (adapter) =>
      adapter.id === "onnx-fp16" && adapter.availability === "available",
  );
  const filtered = models.filter((item) =>
    `${item.descriptor.name} ${item.artifact_id} ${item.descriptor.state.format}`
      .toLowerCase()
      .includes(search.toLowerCase()),
  );

  if (!token)
    return (
      <div className="login-shell">
        <div className="login-story">
          <div className="brand">
            <Logo />
            ModelPort<span className="version">LOCAL / 0.1</span>
          </div>
          <div className="login-copy">
            <div className="eyebrow">YOUR MODELS. YOUR MACHINE.</div>
            <h1>
              Convert models.
              <br />
              Verify the results.
              <br />
              <span>Run with confidence.</span>
            </h1>
            <p>
              A local workspace for model conversion, numerical validation, and
              honest performance measurements.
            </p>
            <div className="login-route">
              <span>PyTorch</span>
              <ArrowRight />
              <span>ONNX</span>
              <ArrowRight />
              <span>CPU runtime</span>
            </div>
          </div>
          <div className="login-foot">
            <ShieldCheck size={17} />
            Model files stay on this machine.
          </div>
        </div>
        <main className="login-form">
          <div className="eyebrow">OPERATOR ACCESS</div>
          <h2>Connect to your workspace</h2>
          <p>Enter your local operator token to open ModelPort.</p>
          <form onSubmit={login}>
            <label htmlFor="token">Operator token</label>
            <input
              id="token"
              type="password"
              autoComplete="off"
              value={draftToken}
              onChange={(event) => setDraftToken(event.target.value)}
              placeholder="Paste your operator token"
              required
            />
            <button className="primary" disabled={loading}>
              {loading ? (
                <LoaderCircle className="spin" size={17} />
              ) : (
                <ArrowRight size={17} />
              )}
              Open workspace
            </button>
          </form>
          {error && (
            <div className="error" role="alert">
              {error}
            </div>
          )}
          <div className="token-help">
            <FileCode2 size={18} />
            <div>
              Find your credential locally<code>modelport token</code>
              <small>
                The credential stays in memory for this browser session.
              </small>
            </div>
          </div>
          <p className="boundary">
            Single operator · Loopback connection · CPU execution
          </p>
        </main>
      </div>
    );

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <Logo />
          ModelPort
        </div>
        <div className="workspace-label">
          <span className="workspace-icon">L</span>
          <div>
            Local workspace<small>Personal environment</small>
          </div>
          <ChevronRight size={14} />
        </div>
        <div className="nav-label">WORKSPACE</div>
        <nav>
          {pages.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={page === id ? "active" : ""}
              onClick={() => setPage(id)}
            >
              <Icon size={18} />
              {label}
              {id === "jobs" && running.length > 0 && (
                <span className="count">{running.length}</span>
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-note">
          <ShieldCheck size={22} />
          <strong>Evidence before confidence</strong>
          <p>Every validation records its policy, inputs, and runtime.</p>
          <span>LOCAL-FIRST BY DESIGN</span>
        </div>
        <div className="sidebar-bottom">
          <span className={`connection-dot ${connected ? "online" : ""}`} />
          {connected ? "Backend connected" : "Backend disconnected"}
          <button
            aria-label="Disconnect workspace"
            title="Disconnect workspace"
            onClick={() => {
              setToken("");
              setConnected(false);
            }}
          >
            <LogOut size={16} />
          </button>
        </div>
      </aside>
      <main className="main">
        <header className="topbar">
          <div>
            Workspace <ChevronRight size={13} />
            <strong>{pages.find((p) => p.id === page)?.label}</strong>
          </div>
          <div className="topbar-right">
            <span className="cpu-pill">
              <Cpu size={13} />
              CPU / LOCAL
            </span>
            <span className="avatar">OP</span>
          </div>
        </header>
        <div className="content">
          <div className="page-heading">
            <div>
              <div className="eyebrow">MODEL WORKSPACE</div>
              <h1>{pages.find((p) => p.id === page)?.label}</h1>
              <p>
                {page === "library"
                  ? "From source model to verified artifact. Every step, accounted for."
                  : page === "jobs"
                    ? "Durable history and live events from your local worker."
                    : page === "benchmarks"
                      ? "Measured execution. Recorded conditions. No assumed speedups."
                      : "What this machine can actually convert and run."}
              </p>
            </div>
            <div className="heading-actions">
              <button
                className="secondary icon-button"
                aria-label="Refresh workspace"
                disabled={!!busy}
                onClick={() => void action("Refreshing", () => refresh())}
              >
                <RefreshCw size={17} />
              </button>
              {page === "library" && (
                <>
                  <button
                    className="secondary"
                    disabled={!!busy || !caps}
                    onClick={() =>
                      void action("Demo · generating fixture", launchDemo)
                    }
                  >
                    <FlaskConical size={16} />
                    Run CPU demo
                  </button>
                  <button
                    className="primary"
                    onClick={() => setUploadOpen(true)}
                  >
                    <Plus size={17} />
                    Import model
                  </button>
                </>
              )}
            </div>
          </div>
          {!connected && (
            <div className="error" role="status">
              Connection lost. Your persisted models and jobs will reappear when
              the backend reconnects.
            </div>
          )}
          {error && (
            <div className="error" role="alert">
              <span>{error}</span>
              <button aria-label="Dismiss error" onClick={() => setError("")}>
                <X size={16} />
              </button>
            </div>
          )}
          {busy && (
            <div className="activity-banner" role="status">
              <LoaderCircle size={17} className="spin" />
              {busy}
              <span>Real worker execution · progress is stage based</span>
            </div>
          )}
          <div className="stats">
            <div>
              <span className="stat-icon blue">
                <Layers3 size={20} />
              </span>
              <div>
                <span>Stored artifacts</span>
                <strong>
                  {models.length}
                  <small>immutable bundles</small>
                </strong>
              </div>
            </div>
            <div>
              <span className="stat-icon teal">
                <ShieldCheck size={20} />
              </span>
              <div>
                <span>Validated artifacts</span>
                <strong>
                  {
                    models.filter((m) => m.validation_state === "validated")
                      .length
                  }
                  <small>on recorded inputs</small>
                </strong>
              </div>
            </div>
            <div>
              <span className="stat-icon violet">
                <Activity size={20} />
              </span>
              <div>
                <span>Active jobs</span>
                <strong>
                  {running.length}
                  <small>
                    {running[0]?.stage.replaceAll("-", " ") ??
                      "worker available"}
                  </small>
                </strong>
              </div>
            </div>
          </div>

          {page === "library" && (
            <>
              <section className="panel library-panel">
                <div className="panel-toolbar">
                  <div>
                    <h2>
                      Model library <span>{models.length}</span>
                    </h2>
                    <p>Source models and derived artifacts</p>
                  </div>
                  <label className="search">
                    <Search size={16} />
                    <input
                      aria-label="Search models"
                      placeholder="Search name, format, or digest…"
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                    />
                  </label>
                </div>
                {models.length === 0 ? (
                  <Empty
                    icon={<Box size={34} />}
                    title="Your first model starts here"
                  >
                    Import ONNX or a registered PyTorch bundle, or run the CPU
                    demo to generate a real fixture.
                  </Empty>
                ) : (
                  <div className="table-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th>Model / artifact</th>
                          <th>Representation</th>
                          <th>Precision</th>
                          <th>Size</th>
                          <th>Evidence</th>
                          <th />
                        </tr>
                      </thead>
                      <tbody>
                        {filtered.map((item) => (
                          <tr
                            key={item.artifact_id}
                            className={
                              selected === item.artifact_id
                                ? "selected-row"
                                : ""
                            }
                          >
                            <td>
                              <button
                                className="model-link"
                                onClick={() => {
                                  setSelected(item.artifact_id);
                                  setTab("inspection");
                                }}
                              >
                                <span
                                  className={`model-icon ${item.descriptor.state.format === "onnx" ? "onnx" : ""}`}
                                >
                                  <Box size={18} />
                                </span>
                                <span>
                                  <strong>{item.descriptor.name}</strong>
                                  <code>{short(item.artifact_id)}…</code>
                                </span>
                              </button>
                            </td>
                            <td>
                              <span className="format-tag">
                                {item.descriptor.state.format ===
                                "pytorch-bundle"
                                  ? "PyTorch bundle"
                                  : item.descriptor.state.format.toUpperCase()}
                              </span>
                              {item.descriptor.state.optimized && (
                                <small className="subtle">
                                  Basic optimized
                                </small>
                              )}
                            </td>
                            <td className="mono">
                              {item.descriptor.state.precision.toUpperCase()}
                            </td>
                            <td className="mono">
                              {bytes(
                                item.files.reduce(
                                  (total, file) => total + file.size_bytes,
                                  0,
                                ),
                              )}
                            </td>
                            <td>
                              <Badge state={item.validation_state} />
                            </td>
                            <td>
                              <button
                                className="table-action"
                                aria-label={`Inspect ${short(item.artifact_id)}`}
                                onClick={() => {
                                  setSelected(item.artifact_id);
                                  setTab("inspection");
                                }}
                              >
                                <ChevronRight size={18} />
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                <div className="panel-footer">
                  <Database size={13} />
                  <span>Content addressed storage · SHA-256 integrity</span>
                  <span>
                    {models.length
                      ? "Persisted locally"
                      : "No external model downloads required"}
                  </span>
                </div>
              </section>
              {model ? (
                <section className="panel detail-panel">
                  <div className="detail-heading">
                    <div>
                      <div className="eyebrow">ARTIFACT DETAILS</div>
                      <h2>{model.descriptor.name}</h2>
                      <code>{model.artifact_id}</code>
                    </div>
                    <button
                      className="secondary"
                      onClick={() =>
                        void action("Downloading artifact", () =>
                          download(
                            token,
                            `/artifacts/${selected}/download`,
                            `modelport-${short(selected)}.zip`,
                          ),
                        )
                      }
                    >
                      <ArrowDownToLine size={16} />
                      Download artifact
                    </button>
                  </div>
                  <div
                    className="tabs"
                    role="tablist"
                    aria-label="Artifact evidence"
                  >
                    {(
                      [
                        "inspection",
                        "convert",
                        "validation",
                        "provenance",
                      ] as Tab[]
                    ).map((item) => (
                      <button
                        role="tab"
                        aria-selected={tab === item}
                        key={item}
                        className={tab === item ? "active" : ""}
                        onClick={() => setTab(item)}
                      >
                        {item === "convert" ? "Convert & optimize" : item}
                      </button>
                    ))}
                  </div>
                  <div className="tab-body">
                    {tab === "inspection" && (
                      <>
                        <div className="section-title">
                          <h3>Tensor signatures</h3>
                          <span>Strict names, shapes, and dtypes</span>
                        </div>
                        <div className="signature-grid">
                          {(["inputs", "outputs"] as const).map((direction) => (
                            <div className="signature-box" key={direction}>
                              <div>
                                {direction.toUpperCase()}
                                <span>
                                  {model.descriptor[direction].length}
                                </span>
                              </div>
                              {model.descriptor[direction].map((spec) => (
                                <div className="signature" key={spec.name}>
                                  <code>{spec.name}</code>
                                  <span className="mono">
                                    [
                                    {spec.dimensions
                                      .map((d) => d ?? "?")
                                      .join(", ")}
                                    ]
                                  </span>
                                  <span>{spec.dtype}</span>
                                </div>
                              ))}
                            </div>
                          ))}
                        </div>
                        <div className="inspection-grid">
                          <div>
                            <h3>Graph operators</h3>
                            <div className="operator-list">
                              {Object.entries(
                                (model.descriptor.metadata.operators ??
                                  {}) as Record<string, number>,
                              ).map(([name, count]) => (
                                <span key={name}>
                                  <code>{name}</code>
                                  <b>{count}</b>
                                </span>
                              ))}
                              {!model.descriptor.metadata.operators && (
                                <p className="muted">
                                  {model.descriptor.state.architecture
                                    ? `Trusted architecture: ${model.descriptor.state.architecture} / v${model.descriptor.state.architecture_version}`
                                    : "Weights only. No executable graph is claimed."}
                                </p>
                              )}
                            </div>
                          </div>
                          <div>
                            <h3>Runtime compatibility</h3>
                            <Badge
                              state={model.descriptor.runtime_compatibility}
                            />
                            <p className="muted">
                              {model.descriptor.state.runtime ??
                                "No runtime binding"}{" "}
                              ·{" "}
                              {model.descriptor.state.provider ??
                                "Not executable"}
                            </p>
                            {model.descriptor.blockers.map((text) => (
                              <p className="warning-text" key={text}>
                                {text}
                              </p>
                            ))}
                            {model.descriptor.warnings.map((text) => (
                              <p className="muted" key={text}>
                                {text}
                              </p>
                            ))}
                          </div>
                        </div>
                        <h3>File inventory</h3>
                        <div className="table-scroll">
                          <table className="compact">
                            <thead>
                              <tr>
                                <th>File</th>
                                <th>Size</th>
                                <th>SHA-256</th>
                              </tr>
                            </thead>
                            <tbody>
                              {model.files.map((file) => (
                                <tr key={file.path}>
                                  <td>
                                    <FileCode2 size={13} /> {file.path}
                                  </td>
                                  <td>{bytes(file.size_bytes)}</td>
                                  <td>
                                    <code title={file.sha256}>
                                      {file.sha256.slice(0, 28)}…
                                    </code>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                        <details>
                          <summary>Complete inspection metadata</summary>
                          <pre>
                            {JSON.stringify(model.descriptor.metadata, null, 2)}
                          </pre>
                        </details>
                      </>
                    )}
                    {tab === "convert" && (
                      <div className="conversion-grid">
                        <div>
                          <h3>Configure a transformation</h3>
                          <p className="muted">
                            Only implemented routes with a passing environment
                            probe are selectable.
                          </p>
                          <label htmlFor="precision">
                            Target representation
                          </label>
                          <select
                            id="precision"
                            value={precision}
                            onChange={(e) => {
                              setPrecision(e.target.value);
                              setPlan(null);
                            }}
                          >
                            <option value="fp32">ONNX · FP32</option>
                            <option
                              value="dynamic-int8"
                              disabled={!canQuantize}
                            >
                              ONNX · Dynamic INT8
                              {!canQuantize ? " — probe unavailable" : ""}
                            </option>
                            <option
                              value="static-int8"
                              disabled={!canStaticQuantize}
                            >
                              ONNX · Static INT8
                              {!canStaticQuantize ? " — probe unavailable" : ""}
                            </option>
                            <option value="fp16" disabled={!canFloat16}>
                              ONNX · FP16 weights / FP32 I/O
                              {!canFloat16 ? " — probe unavailable" : ""}
                            </option>
                          </select>
                          {precision === "static-int8" && (
                            <fieldset className="dataset-panel">
                              <legend>
                                Calibration and held-out validation
                              </legend>
                              <p className="muted">
                                Upload separate named-tensor NPZ files.
                                Identical samples across the two datasets are
                                rejected. Calibration quality affects numerical
                                fidelity.
                              </p>
                              {(["calibration", "validation"] as const).map(
                                (purpose) => (
                                  <div key={purpose}>
                                    <label htmlFor={`${purpose}-dataset`}>
                                      {purpose === "calibration"
                                        ? "Calibration dataset"
                                        : "Validation dataset"}
                                    </label>
                                    <select
                                      id={`${purpose}-dataset`}
                                      value={
                                        purpose === "calibration"
                                          ? calibrationId
                                          : validationId
                                      }
                                      onChange={(event) => {
                                        (purpose === "calibration"
                                          ? setCalibrationId
                                          : setValidationId)(
                                          event.target.value,
                                        );
                                        setPlan(null);
                                      }}
                                    >
                                      <option value="">
                                        Select a registered dataset
                                      </option>
                                      {datasets
                                        .filter(
                                          (item) => item.purpose === purpose,
                                        )
                                        .map((item) => (
                                          <option
                                            key={item.dataset_id}
                                            value={item.dataset_id}
                                          >
                                            {item.sample_count} samples ·{" "}
                                            {short(item.dataset_id)}
                                          </option>
                                        ))}
                                    </select>
                                    <label htmlFor={`${purpose}-upload`}>
                                      Upload {purpose} NPZ
                                    </label>
                                    <input
                                      id={`${purpose}-upload`}
                                      type="file"
                                      accept=".npz"
                                      disabled={!!busy}
                                      onChange={(event) => {
                                        const file = event.target.files?.[0];
                                        if (file)
                                          void action(
                                            `Inspecting ${purpose} tensors`,
                                            () => importDataset(file, purpose),
                                          );
                                        event.target.value = "";
                                      }}
                                    />
                                  </div>
                                ),
                              )}
                            </fieldset>
                          )}
                          {precision === "fp16" && (
                            <p className="muted">
                              FP16 graph weights, with FP32 inputs and outputs.
                              CPU kernels may promote arithmetic to FP32;
                              speedup is not assumed.
                            </p>
                          )}
                          <label className="checkbox">
                            <input
                              type="checkbox"
                              checked={optimize}
                              disabled={!canOptimize}
                              onChange={(e) => {
                                setOptimize(e.target.checked);
                                setPlan(null);
                              }}
                            />
                            Apply ONNX basic graph optimization
                          </label>
                          <div className="config-note">
                            <Cpu size={16} />
                            <div>
                              CPUExecutionProvider
                              <small>
                                No implicit device or precision fallback.
                              </small>
                            </div>
                          </div>
                          <button
                            className="primary"
                            disabled={
                              !!busy ||
                              !caps ||
                              model.descriptor.runtime_compatibility ===
                                "blocked"
                            }
                            onClick={() =>
                              void action("Planning route", async () =>
                                setPlan(
                                  await api<Plan>(token, "/plans", {
                                    source_id: selected,
                                    precision,
                                    optimize,
                                    calibration_id:
                                      precision === "static-int8"
                                        ? calibrationId || null
                                        : null,
                                    validation_id:
                                      precision === "static-int8"
                                        ? validationId || null
                                        : null,
                                  }),
                                ),
                              )
                            }
                          >
                            <SlidersHorizontal size={16} />
                            Preview route
                          </button>
                        </div>
                        <div className="route-preview">
                          <div className="eyebrow">EXECUTION PLAN</div>
                          {plan ? (
                            <>
                              <h3>
                                {plan.steps.length} verified{" "}
                                {plan.steps.length === 1
                                  ? "adapter"
                                  : "adapters"}
                              </h3>
                              <div className="route-steps">
                                {plan.steps.map((step, i) => (
                                  <div key={step.adapter_id}>
                                    <span>{i + 1}</span>
                                    <div>
                                      <strong>{step.adapter_id}</strong>
                                      <small>
                                        v{step.version} ·{" "}
                                        {step.target.precision} · CPU
                                      </small>
                                    </div>
                                    <Check size={16} />
                                  </div>
                                ))}
                              </div>
                              <p className="muted">Checks performed now</p>
                              <ul>
                                {plan.preflight.checked.map((item) => (
                                  <li key={item}>{item}</li>
                                ))}
                              </ul>
                              <details>
                                <summary>
                                  Restrictions and remaining probes
                                </summary>
                                <ul>
                                  {[
                                    ...plan.preflight.warnings,
                                    ...plan.preflight.unresolved,
                                  ].map((item) => (
                                    <li key={item}>{item}</li>
                                  ))}
                                </ul>
                              </details>
                              <button
                                className="primary full-width"
                                disabled={!!busy}
                                onClick={() =>
                                  void action(
                                    "Converting and validating",
                                    async () => {
                                      const done = await waitJob(
                                        await api<Job>(
                                          token,
                                          "/conversions",
                                          plan,
                                        ),
                                      );
                                      setSelected(
                                        String(done.result?.artifact_id),
                                      );
                                      setTab("validation");
                                    },
                                  )
                                }
                              >
                                <Play size={15} />
                                Convert & validate
                              </button>
                            </>
                          ) : (
                            <Empty
                              icon={<ArrowLeftRight size={28} />}
                              title="Review before you run"
                            >
                              Preview the exact adapters, normalized options,
                              and remaining runtime checks.
                            </Empty>
                          )}
                        </div>
                      </div>
                    )}
                    {tab === "validation" && (
                      <>
                        <div className="section-title">
                          <div>
                            <h3>Numerical validation</h3>
                            <p className="muted">
                              One failed required output fails the recorded run.
                            </p>
                          </div>
                          {!!model.manifest.reference_id && (
                            <button
                              className="secondary"
                              disabled={!!busy}
                              onClick={() =>
                                void action("Revalidating", async () => {
                                  await waitJob(
                                    await api<Job>(token, "/validations", {
                                      source_id: model.manifest.reference_id,
                                      target_id: selected,
                                      policy:
                                        model.manifest.validation_policy ??
                                        "fp32-default",
                                      validation_id:
                                        (
                                          model.manifest.requested_options as
                                            | Record<string, unknown>
                                            | undefined
                                        )?.validation_id ?? null,
                                    }),
                                  );
                                })
                              }
                            >
                              <RefreshCw size={15} />
                              Revalidate
                            </button>
                          )}
                        </div>
                        {selectedReports.length === 0 ? (
                          <Empty
                            icon={<ShieldCheck size={30} />}
                            title="No numerical evidence yet"
                          >
                            This artifact is unverified. A structural import or
                            successful file write does not establish numerical
                            correctness.
                          </Empty>
                        ) : (
                          selectedReports.map((report) => (
                            <article
                              className="validation-card"
                              key={report.report_id}
                            >
                              <div className="validation-top">
                                <Badge state={report.state} />
                                <code>
                                  {report.policy.name} / v
                                  {report.policy.version}
                                </code>
                                <small>
                                  {new Date(report.created_at).toLocaleString()}
                                </small>
                              </div>
                              <p>{report.evidence_scope}</p>
                              <div className="evidence-facts">
                                <span>
                                  <strong>{report.dataset.sample_count}</strong>
                                  samples
                                </span>
                                <span>
                                  <strong>{report.dataset.batch_count}</strong>
                                  batches
                                </span>
                                <span>
                                  <strong>{report.policy.gate}</strong>required
                                  gate
                                </span>
                                <span>
                                  <strong>{report.dataset.kind}</strong>input
                                  scope
                                </span>
                              </div>
                              <div className="table-scroll">
                                <table className="compact">
                                  <thead>
                                    <tr>
                                      <th>Output / batch</th>
                                      <th>Max abs. error</th>
                                      <th>Normalized L2</th>
                                      <th>Cosine</th>
                                      <th>Top-1 agreement</th>
                                      <th>Gate</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {report.outputs.map((row) => (
                                      <tr
                                        key={`${row.output}-${row.batch_index}`}
                                      >
                                        <td>
                                          <code>{row.output}</code> /{" "}
                                          {row.batch_index}
                                        </td>
                                        <td className="mono">
                                          {number(row.max_absolute_error, 8)}
                                        </td>
                                        <td className="mono">
                                          {number(row.normalized_l2, 6)}
                                        </td>
                                        <td className="mono">
                                          {number(row.cosine_similarity, 6)}
                                        </td>
                                        <td>
                                          {row.top1_agreement == null
                                            ? "—"
                                            : `${number(row.top1_agreement * 100, 1)}%`}
                                        </td>
                                        <td>
                                          <Badge
                                            state={
                                              row.passed ? "passed" : "failed"
                                            }
                                          />
                                        </td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                              {report.failures.map((failure) => (
                                <p className="warning-text" key={failure}>
                                  {failure}
                                </p>
                              ))}
                              <details>
                                <summary>
                                  Policy, digests and complete metrics
                                </summary>
                                <pre>{JSON.stringify(report, null, 2)}</pre>
                              </details>
                            </article>
                          ))
                        )}
                      </>
                    )}
                    {tab === "provenance" && (
                      <>
                        <h3>Immutable conversion manifest</h3>
                        <p className="muted">
                          The bundle digest includes immutable model files.
                          Later validation and benchmark records reference that
                          digest independently.
                        </p>
                        <pre>{JSON.stringify(model.manifest, null, 2)}</pre>
                      </>
                    )}
                  </div>
                </section>
              ) : (
                <div className="hint-card">
                  <CircleHelp size={19} />
                  <div>
                    <strong>Inspect first. Convert with evidence.</strong>
                    <p>
                      Select an artifact to explore its signatures, plan a
                      route, and review numerical results.
                    </p>
                  </div>
                  <span>01 / IMPORT → 02 / CONVERT → 03 / VERIFY</span>
                </div>
              )}
            </>
          )}

          {page === "jobs" && (
            <section className="panel">
              <div className="panel-toolbar">
                <div>
                  <h2>Execution history</h2>
                  <p>
                    Attempts and important state events survive service
                    restarts.
                  </p>
                </div>
                <span className="live-label">
                  <Radio size={14} />
                  LIVE EVENTS
                </span>
              </div>
              {jobs.length === 0 ? (
                <Empty icon={<Activity size={32} />} title="No jobs yet">
                  Import a model or run the CPU demo to start a real workflow.
                </Empty>
              ) : (
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Job</th>
                        <th>State</th>
                        <th>Current stage</th>
                        <th>Attempt</th>
                        <th>Created</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {jobs.map((job) => (
                        <tr key={job.job_id}>
                          <td>
                            <button
                              className="text-button"
                              onClick={() => setActiveJob(job.job_id)}
                            >
                              {job.kind} <code>{short(job.job_id)}</code>
                            </button>
                          </td>
                          <td>
                            <Badge state={job.state} />
                          </td>
                          <td>{job.stage}</td>
                          <td>{job.attempt}</td>
                          <td className="nowrap">
                            {new Date(job.created_at).toLocaleTimeString()}
                          </td>
                          <td>
                            {!terminal(job.state) ? (
                              <button
                                className="secondary small"
                                onClick={() =>
                                  void action(
                                    "Requesting cancellation",
                                    async () => {
                                      await api(
                                        token,
                                        `/jobs/${job.job_id}/cancel`,
                                        {},
                                      );
                                    },
                                  )
                                }
                              >
                                Cancel
                              </button>
                            ) : job.state !== "succeeded" ? (
                              <button
                                className="secondary small"
                                onClick={() =>
                                  void action("Submitting retry", async () => {
                                    setActiveJob(
                                      (
                                        await api<Job>(
                                          token,
                                          `/jobs/${job.job_id}/retry`,
                                          {},
                                        )
                                      ).job_id,
                                    );
                                  })
                                }
                              >
                                Retry
                              </button>
                            ) : (
                              <span className="muted">Complete</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {activeJob && (
                <div className="job-detail">
                  <div className="section-title">
                    <h3>Job {short(activeJob)}</h3>
                    <button
                      className="secondary small"
                      onClick={() =>
                        void action("Loading logs", async () =>
                          setLogs(
                            (
                              await api<{ text: string }>(
                                token,
                                `/jobs/${activeJob}/logs`,
                              )
                            ).text,
                          ),
                        )
                      }
                    >
                      View bounded log
                    </button>
                  </div>
                  <div className="event-stream">
                    {events.map((event, i) => (
                      <span key={`${event}-${i}`}>
                        <span className="event-index">{i + 1}</span>
                        {event}
                      </span>
                    ))}
                  </div>
                  {jobs.find((j) => j.job_id === activeJob)?.error && (
                    <pre className="error-text">
                      {JSON.stringify(
                        jobs.find((j) => j.job_id === activeJob)?.error,
                        null,
                        2,
                      )}
                    </pre>
                  )}
                  {logs && <pre>{logs}</pre>}
                </div>
              )}
            </section>
          )}

          {page === "benchmarks" && (
            <>
              <section className="panel benchmark-config">
                <div>
                  <h2>Measure an artifact</h2>
                  <p>
                    Fresh worker process · sequential CPU execution · one thread
                  </p>
                </div>
                <div className="benchmark-form">
                  <label>
                    Artifact
                    <select
                      aria-label="Benchmark artifact"
                      value={selected}
                      onChange={(e) => setSelected(e.target.value)}
                    >
                      <option value="">Choose artifact</option>
                      {models
                        .filter(
                          (m) =>
                            m.descriptor.runtime_compatibility !== "blocked" &&
                            m.validation_state !== "failed",
                        )
                        .map((m) => (
                          <option key={m.artifact_id} value={m.artifact_id}>
                            {m.descriptor.name} · {short(m.artifact_id)}
                          </option>
                        ))}
                    </select>
                  </label>
                  <label>
                    Batch
                    <input
                      type="number"
                      min={1}
                      max={64}
                      value={batch}
                      onChange={(e) => setBatch(Number(e.target.value))}
                    />
                  </label>
                  <label>
                    Iterations
                    <input
                      type="number"
                      min={2}
                      max={1000}
                      value={iterations}
                      onChange={(e) => setIterations(Number(e.target.value))}
                    />
                  </label>
                  <button
                    className="primary"
                    disabled={!selected || !!busy}
                    onClick={() =>
                      void action("Benchmarking actual inference", async () => {
                        await waitJob(
                          await api<Job>(token, "/benchmarks", {
                            artifact_id: selected,
                            config: {
                              batch_size: batch,
                              iterations,
                              threads: 1,
                              warmup: 5,
                            },
                          }),
                        );
                      })
                    }
                  >
                    <Play size={15} />
                    Run benchmark
                  </button>
                </div>
              </section>
              <section className="panel">
                <div className="panel-toolbar">
                  <div>
                    <h2>Latency comparison</h2>
                    <p>
                      Mean warm inference latency · milliseconds · lower is
                      faster
                    </p>
                  </div>
                  <span className="format-tag">ACTUAL MEASUREMENTS</span>
                </div>
                {benchmarks.length === 0 ? (
                  <Empty
                    icon={<ArrowLeftRight size={32} />}
                    title="No measurements yet"
                  >
                    Run a benchmark to record latency, throughput, load time and
                    process memory.
                  </Empty>
                ) : (
                  <>
                    <div className="benchmark-chart">
                      {benchmarks.slice(0, 8).map((report) => (
                        <div className="chart-row" key={report.report_id}>
                          <div>
                            <strong>
                              {models.find(
                                (m) => m.artifact_id === report.artifact_id,
                              )?.descriptor.state.precision ?? "artifact"}
                            </strong>
                            <code>{short(report.artifact_id)}</code>
                          </div>
                          <svg
                            viewBox="0 0 600 26"
                            role="img"
                            aria-label={`${number(report.measurements.mean_ms)} milliseconds`}
                          >
                            <rect
                              width="600"
                              height="26"
                              rx="4"
                              fill="#f0f4f8"
                            />
                            <rect
                              width={Math.max(
                                1,
                                (report.measurements.mean_ms /
                                  Math.max(
                                    ...benchmarks
                                      .slice(0, 8)
                                      .map((b) => b.measurements.mean_ms),
                                  )) *
                                  600,
                              )}
                              height="26"
                              rx="4"
                              fill="#247f88"
                            />
                          </svg>
                          <strong>
                            {number(report.measurements.mean_ms)}{" "}
                            <small>ms</small>
                          </strong>
                        </div>
                      ))}
                    </div>
                    <div className="comparison-note">
                      <CircleHelp size={16} />
                      Compare matching batch sizes, thread settings and
                      environments. Tail percentiles below 100 samples have
                      limited evidence.
                    </div>
                    <div className="table-scroll">
                      <table>
                        <thead>
                          <tr>
                            <th>Artifact / runtime</th>
                            <th>Batch / threads</th>
                            <th>Samples</th>
                            <th>P50 · ms</th>
                            <th>P95 · ms</th>
                            <th>Items / s</th>
                            <th>Evidence</th>
                            <th>Report</th>
                          </tr>
                        </thead>
                        <tbody>
                          {benchmarks.map((report) => (
                            <tr key={report.report_id}>
                              <td>
                                <code>{short(report.artifact_id)}</code>
                                <small className="subtle">
                                  {report.runtime.runtime}
                                </small>
                              </td>
                              <td>
                                {report.config.batch_size} /{" "}
                                {report.config.threads}
                              </td>
                              <td>{report.measurements.sample_count}</td>
                              <td className="mono">
                                {number(report.measurements.p50_ms)}
                              </td>
                              <td className="mono">
                                {number(report.measurements.p95_ms)}
                              </td>
                              <td className="mono">
                                {number(
                                  report.measurements
                                    .throughput_items_per_second,
                                  0,
                                )}
                              </td>
                              <td>
                                <Badge state={report.validation_state} />
                              </td>
                              <td>
                                <button
                                  className="text-button"
                                  onClick={() =>
                                    void action("Downloading report", () =>
                                      download(
                                        token,
                                        `/benchmarks/${report.report_id}/export?format=html`,
                                        "benchmark.html",
                                      ),
                                    )
                                  }
                                >
                                  HTML <ArrowDownToLine size={13} />
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    {benchmarks.slice(0, 3).map((report) => (
                      <details
                        className="benchmark-details"
                        key={report.report_id}
                      >
                        <summary>
                          {short(report.artifact_id)} · load{" "}
                          {number(report.measurements.load_ms, 1)} ms · sampled
                          RSS{" "}
                          {bytes(report.measurements.sampled_peak_rss_bytes)} ·
                          configuration and provenance
                        </summary>
                        <pre>{JSON.stringify(report, null, 2)}</pre>
                      </details>
                    ))}
                  </>
                )}
              </section>
            </>
          )}

          {page === "environment" && (
            <>
              <section className="panel">
                <div className="panel-toolbar">
                  <div>
                    <h2>Executable capabilities</h2>
                    <p>
                      Implementation, environment availability, and model
                      compatibility are separate checks.
                    </p>
                  </div>
                  <button
                    className="secondary"
                    disabled={!!busy}
                    onClick={() =>
                      void action(
                        "Running real capability probes",
                        async () => {
                          await waitJob(await api<Job>(token, "/doctor", {}));
                        },
                      )
                    }
                  >
                    <RefreshCw size={16} />
                    Run doctor
                  </button>
                </div>
                {caps ? (
                  <>
                    <div className="environment-grid">
                      <div>
                        <Cpu size={22} />
                        <h3>CPU runtime</h3>
                        <p>
                          {String(caps.environment.logical_cores)} logical cores
                          · {bytes(Number(caps.environment.ram_bytes))} RAM
                        </p>
                        <small>{String(caps.environment.os)}</small>
                      </div>
                      <div>
                        <ShieldCheck size={22} />
                        <h3>Verified execution</h3>
                        <p>
                          {caps.adapters.filter((a) => a.probe_verified).length}{" "}
                          adapters passed real probes
                        </p>
                        <small>
                          Checked {new Date(caps.checked_at).toLocaleString()}
                        </small>
                      </div>
                    </div>
                    <div className="table-scroll">
                      <table>
                        <thead>
                          <tr>
                            <th>Adapter</th>
                            <th>Implementation</th>
                            <th>Environment</th>
                            <th>Model compatibility</th>
                          </tr>
                        </thead>
                        <tbody>
                          {caps.adapters.map((adapter) => (
                            <tr key={String(adapter.id)}>
                              <td>
                                <strong>{String(adapter.id)}</strong>
                                <small className="subtle">
                                  {String(
                                    adapter.reason ?? adapter.target ?? "",
                                  )}
                                </small>
                              </td>
                              <td>{String(adapter.implementation)}</td>
                              <td>
                                <Badge state={String(adapter.availability)} />
                              </td>
                              <td>
                                {adapter.implementation === "planned"
                                  ? "No executable edge"
                                  : "Checked for each model"}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div className="package-grid">
                      {Object.entries(caps.packages).map(([name, version]) => (
                        <div key={name}>
                          <span>{name}</span>
                          <code>{version ?? "missing"}</code>
                        </div>
                      ))}
                    </div>
                    <details className="benchmark-details">
                      <summary>
                        Complete probe evidence and restrictions
                      </summary>
                      <pre>{JSON.stringify(caps, null, 2)}</pre>
                    </details>
                  </>
                ) : (
                  <Empty
                    icon={<Cpu size={30} />}
                    title="Capability probes pending"
                  >
                    Run doctor or wait for startup probes. Targets become
                    selectable only after actual execution succeeds.
                  </Empty>
                )}
              </section>
              <div className="hint-card">
                <ShieldCheck size={20} />
                <div>
                  <strong>Local single-operator security boundary</strong>
                  <p>
                    SafeTensors and process limits reduce risk. Native runtimes
                    are not a security sandbox. Remote imports require public
                    HTTPS and file checksums. Uploaded code is disabled.
                  </p>
                </div>
              </div>
            </>
          )}
          <footer className="page-footer">
            <span>
              ModelPort <b>0.1</b>
            </span>
            <span>
              Convert models. Verify the results. Run with confidence.
            </span>
            <span>
              <span className="tiny-dot" />
              Local CPU workspace
            </span>
          </footer>
        </div>
      </main>
      {uploadOpen && (
        <div
          className="modal-backdrop"
          onClick={(e) => {
            if (e.target === e.currentTarget && !busy) setUploadOpen(false);
          }}
        >
          <section
            ref={modalRef}
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="upload-title"
          >
            <div className="section-title">
              <h2 id="upload-title">Import a model</h2>
              <button
                aria-label="Close import"
                className="icon-button secondary"
                disabled={!!busy}
                onClick={() => setUploadOpen(false)}
              >
                <X size={17} />
              </button>
            </div>
            <label className="import-source">
              Source
              <select
                aria-label="Source"
                value={importSource}
                onChange={(e) => setImportSource(e.target.value)}
              >
                <option value="local">Local model files</option>
                <option value="https">Public HTTPS</option>
                <option value="huggingface">Hugging Face repository</option>
              </select>
            </label>
            {importSource === "local" ? (
              <>
                <p>
                  Choose an ONNX model, SafeTensors weights, or both files of a
                  registered PyTorch bundle.
                </p>
                <label className="dropzone">
                  <Upload size={32} />
                  <strong>Choose model files</strong>
                  <span>ONNX · SafeTensors · bundle.json · external .data</span>
                  <input
                    aria-label="Model files"
                    type="file"
                    multiple
                    accept=".onnx,.safetensors,.json,.data"
                    onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
                  />
                </label>
                <p className="muted">
                  128 MiB per file · 256 MiB total · 64 files maximum. Archives
                  and executable model files are rejected.
                </p>
                <ul className="upload-files">
                  {files.map((file) => (
                    <li key={file.name}>
                      <FileCode2 size={15} />
                      {file.name}
                      <span>{bytes(file.size)}</span>
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <>
                <p>
                  Paste a download manifest with an explicit SHA-256 for every
                  file. Hugging Face imports require the full 40-character
                  commit revision.
                </p>
                <label className="import-source">
                  Import manifest (JSON)
                  <textarea
                    rows={10}
                    spellCheck={false}
                    value={importManifest}
                    onChange={(e) => setImportManifest(e.target.value)}
                    placeholder={
                      importSource === "https"
                        ? '{"files":[{"url":"https://example.org/model.onnx","path":"model.onnx","sha256":"..."}]}'
                        : '{"repository":"owner/model","revision":"...","files":[{"path":"onnx/model.onnx","destination":"model.onnx","sha256":"..."}]}'
                    }
                  />
                </label>
                <p className="muted">
                  Public repositories only. Credentials, private network
                  destinations, archives and executable model files are
                  rejected. Downloads stay on this machine.
                </p>
              </>
            )}
            {error && (
              <div className="error" role="alert">
                {error}
              </div>
            )}
            <div className="modal-actions">
              <button
                className="secondary"
                disabled={!!busy}
                onClick={() => setUploadOpen(false)}
              >
                Cancel
              </button>
              <button
                className="primary"
                disabled={
                  (importSource === "local"
                    ? !files.length
                    : !importManifest.trim()) || !!busy
                }
                onClick={() =>
                  void action("Uploading and inspecting model", importFiles)
                }
              >
                {busy ? (
                  <LoaderCircle className="spin" size={16} />
                ) : (
                  <Plus size={16} />
                )}
                Import & inspect
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
