import { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import registry from "../../shared/symptoms.json";
import samples from "../../shared/samples.json";
import { examples } from "./Home";
import { Icon } from "./Icon";
import { emptySession, usePreferences, useSession, type Report } from "./state";
import { API, request } from "./api";
import type {
  Assessment,
  Extraction,
  Finding,
  Explanation,
  Followup,
} from "./types";
const label = (id: string) =>
  registry.symptoms.find((s) => s.id === id)?.label || id;
const fictionalFindings: Finding[] = [
  {
    test_name: "Hemoglobin",
    value: "13.8",
    unit: "g/dL",
    reference_range: "12.0–16.0",
    source_page: 1,
    method: "fictional",
    needs_review: false,
  },
  {
    test_name: "Glucose",
    value: "92",
    unit: "mg/dL",
    reference_range: "70–99",
    source_page: 1,
    method: "fictional",
    needs_review: false,
  },
  {
    test_name: "Platelets",
    value: "250",
    unit: "×10³/µL",
    reference_range: null,
    source_page: 1,
    method: "fictional",
    needs_review: true,
  },
];
export default function Workspace() {
  const { session, setSession } = useSession();
  const { motion, setMotion } = usePreferences();
  const [params, setParams] = useSearchParams();
  const tab = ["symptoms", "reports", "summary"].includes(
    params.get("tab") || "",
  )
    ? params.get("tab")!
    : "symptoms";
  const [query, setQuery] = useState("");
  const [browseAll, setBrowseAll] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState<
    "checking" | "ready" | "unavailable" | "degraded"
  >(API ? "checking" : "unavailable");
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [explaining, setExplaining] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [height, setHeight] = useState("");
  const [weight, setWeight] = useState("");
  const controller = useRef<AbortController | null>(null);
  const extractionController = useRef<AbortController | null>(null);
  const [extracting, setExtracting] = useState(false);
  const sampleLoaded = useRef("");
  const fileRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    let alive = true;
    if (API)
      request<{ status: "ready" | "degraded" }>("/health", {}, 40000)
        .then((x) => {
          if (alive) setStatus(x.status);
        })
        .catch(() => {
          if (alive) setStatus("unavailable");
        });
    return () => {
      alive = false;
      controller.current?.abort();
      extractionController.current?.abort();
    };
  }, []);
  useEffect(() => {
    const id = params.get("sample");
    if (id && sampleLoaded.current !== id) {
      sampleLoaded.current = id;
      loadSample(id);
    }
  }, [params]);
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [tab]);
  function changeTab(value: string) {
    setParams({ tab: value });
    setError("");
  }
  function loadSample(id: string) {
    const sample = samples.find((s) => s.id === id);
    if (!sample) return;
    controller.current?.abort();
    extractionController.current?.abort();
    setSession({
      ...emptySession,
      sampleId: id,
      symptoms: sample.symptoms,
      result: sample.assessment as Assessment,
      followups: [
        { question_id: "duration", answer: sample.duration, confirmed: false },
      ],
    });
    setExplanation(sample.explanation as Explanation);
    setError("");
    setQuery("");
  }
  function invalidate(update: Partial<typeof session>) {
    setSession((s) => ({ ...s, ...update, result: null, sampleId: null }));
    setExplanation(null);
    setError("");
  }
  function toggle(id: string) {
    invalidate({
      symptoms: session.symptoms.includes(id)
        ? session.symptoms.filter((x) => x !== id)
        : [...session.symptoms, id],
    });
  }
  function followup(id: string, answer: string, confirmed = false) {
    const item: Followup = { question_id: id, answer, confirmed };
    invalidate({
      followups: [
        ...session.followups.filter((f) => f.question_id !== id),
        item,
      ],
    });
  }
  async function assess() {
    setError("");
    setBusy(true);
    setExplanation(null);
    controller.current = new AbortController();
    try {
      const result = await request<Assessment>(
        "/assessments",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            symptoms: session.symptoms,
            followups: session.followups,
          }),
          signal: controller.current.signal,
        },
        40000,
      );
      setSession((s) => ({ ...s, result, sampleId: null }));
      setStatus("ready");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function enhanceSample() {
    if (!session.sampleId) return;
    const id = session.sampleId;
    setExplaining(true);
    try {
      const response = await request<Explanation>(`/samples/${id}/explain`, {
        method: "POST",
      });
      setExplanation(response);
    } catch {
      setExplanation(
        samples.find((s) => s.id === id)!.explanation as Explanation,
      );
    } finally {
      setExplaining(false);
    }
  }
  function updateReport(id: string, patch: Partial<Report>) {
    setSession((s) => ({
      ...s,
      reports: s.reports.map((r) => (r.id === id ? { ...r, ...patch } : r)),
    }));
  }
  async function processFiles(files: FileList | null) {
    if (!files?.length) return;
    setError("");
    const list = Array.from(files);
    if (session.reports.length + list.length > 3) {
      setError(
        "Keep up to three reports in a session. Remove a report before adding more.",
      );
      return;
    }
    const queue = list.map((file) => ({ file, id: crypto.randomUUID() }));
    setSession((s) => ({
      ...s,
      reports: [
        ...s.reports,
        ...queue.map(({ file, id }) => ({
          id,
          name: file.name,
          status: "queued" as const,
          findings: [],
          warnings: [],
        })),
      ],
    }));
    setExtracting(true);
    extractionController.current = new AbortController();
    const signal = extractionController.current.signal;
    for (const { file, id } of queue) {
      if (signal.aborted) {
        updateReport(id, { status: "cancelled" });
        continue;
      }
      if (file.size > 10 * 1024 * 1024) {
        updateReport(id, {
          status: "error",
          error: "This file exceeds 10 MB.",
        });
        continue;
      }
      if (!/\.(pdf|jpe?g|png)$/i.test(file.name)) {
        updateReport(id, {
          status: "error",
          error: "Choose a PDF, JPG or PNG.",
        });
        continue;
      }
      updateReport(id, { status: "extracting" });
      const body = new FormData();
      body.append("file", file);
      try {
        const result = await request<Extraction>(
          "/lab-reports/extract",
          { method: "POST", body, signal },
          115000,
        );
        updateReport(id, {
          status: "done",
          findings: result.findings,
          warnings: result.warnings,
        });
      } catch (e) {
        updateReport(id, {
          status: signal.aborted ? "cancelled" : "error",
          error: (e as Error).message,
        });
      }
    }
    setExtracting(false);
    if (fileRef.current) fileRef.current.value = "";
  }
  function editFinding(
    report: Report,
    index: number,
    key: keyof Finding,
    value: string,
  ) {
    updateReport(report.id, {
      findings: report.findings.map((f, i) =>
        i === index
          ? {
              ...f,
              [key]: key === "test_name" ? value : value || null,
              needs_review: true,
            }
          : f,
      ),
    });
  }
  async function download() {
    setExporting(true);
    try {
      const { createSummary } = await import("./export");
      createSummary(session).save("the-shape-of-health-summary.pdf");
    } catch {
      setError("The PDF could not be created. Please try again.");
    } finally {
      setExporting(false);
    }
  }
  const filtered = registry.symptoms.filter((s) =>
    [s.label, s.id, ...s.aliases].some((v) =>
      v.toLowerCase().replace(/_/g, " ").includes(query.toLowerCase().trim()),
    ),
  );
  const common = [
    "headache",
    "fatigue",
    "high_fever",
    "cough",
    "nausea",
    "skin_rash",
    "itching",
    "breathlessness",
    "runny_nose",
    "abdominal_pain",
    "dizziness",
    "joint_pain",
  ];
  const visibleSymptoms =
    query.trim() || browseAll
      ? filtered
      : filtered.filter((s) => common.includes(s.id));
  const selectedCount = new Set([
    ...session.symptoms,
    ...session.followups
      .filter((f) => f.answer === "Yes" && f.confirmed)
      .map(
        (f) => registry.questions.find((q) => q.id === f.question_id)?.suggests,
      )
      .filter((x): x is string => !!x),
  ]).size;
  const bmi =
    +height >= 80 && +height <= 250 && +weight >= 20 && +weight <= 400
      ? (+weight / (+height / 100) ** 2).toFixed(1)
      : null;
  return (
    <main id="main" className="workspace">
      <div className="workspace-heading">
        <div>
          <span className="eyebrow">YOUR HEALTH TOOLS</span>
          <h1>Your health workspace</h1>
        </div>
        <p>
          Start with symptoms or a report.
          <br /> Review everything in one place.
        </p>
      </div>
      <div className={`connection ${status}`} role="status">
        <i />
        {status === "checking"
          ? "Connecting to the backend. A sleeping service can take a moment."
          : status === "ready"
            ? "Live tools connected."
            : status === "degraded"
              ? "The service is reachable, but the model is not ready. Fictional examples are available."
              : "Live tools are currently unavailable. All three fictional walkthroughs still work."}
      </div>
      <div
        className="workspace-tabs"
        role="tablist"
        aria-label="Workspace areas"
      >
        {["symptoms", "reports", "summary"].map((name, i) => (
          <button
            id={`tab-${name}`}
            role="tab"
            aria-selected={tab === name}
            aria-controls={`panel-${name}`}
            key={name}
            tabIndex={tab === name ? 0 : -1}
            onKeyDown={(e) => {
              if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
                const next = ["symptoms", "reports", "summary"][
                  (i + (e.key === "ArrowRight" ? 1 : 2)) % 3
                ];
                changeTab(next);
                document.getElementById(`tab-${next}`)?.focus();
              }
            }}
            onClick={() => changeTab(name)}
          >
            <small>0{i + 1}</small>
            {name[0].toUpperCase() + name.slice(1)}
          </button>
        ))}
        <button
          className="clear-session"
          disabled={busy || extracting}
          onClick={() => {
            setSession({ ...emptySession });
            setExplanation(null);
            setError("");
            setParams({ tab });
            sampleLoaded.current = "";
          }}
        >
          Clear session ↺
        </button>
      </div>
      {error && (
        <div className="message error" role="alert">
          {error}
        </div>
      )}
      <section
        id={`panel-${tab}`}
        role="tabpanel"
        aria-labelledby={`tab-${tab}`}
      >
        {tab === "symptoms" && (
          <>
            <details className="example-picker">
              <summary>Just exploring? Try a fictional example</summary>
              <div className="sample-strip">
                <span>Choose an example</span>
                {samples.map((s, i) => (
                  <button
                    disabled={busy || extracting}
                    key={s.id}
                    aria-pressed={session.sampleId === s.id}
                    onClick={() => loadSample(s.id)}
                  >
                    <small>0{i + 1}</small>
                    {examples.find((e) => e.id === s.id)?.title || s.name}
                    <span>↗</span>
                  </button>
                ))}
              </div>
            </details>
            <div className="workspace-grid">
              <div className="input-column">
                <div className="section-heading">
                  <h2>What have you noticed?</h2>
                  <span>{selectedCount} selected</span>
                </div>
                <p className="muted">
                  Choose at least two symptoms. Live inputs go to this project’s
                  backend. <Link to="/privacy">How processing works ↗</Link>
                </p>
                <label className="search-label" htmlFor="symptom-search">
                  Search symptoms
                </label>
                <div className="search-box">
                  <span aria-hidden="true">⌕</span>
                  <input
                    disabled={busy}
                    id="symptom-search"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Try headache or fever…"
                    autoComplete="off"
                  />
                  <span>{registry.symptoms.length}</span>
                </div>
                <div
                  className="selected-symptoms"
                  aria-label="Selected symptoms"
                >
                  {session.symptoms.map((id) => (
                    <button
                      disabled={busy}
                      key={id}
                      onClick={() => toggle(id)}
                      aria-label={`Remove ${label(id)}`}
                    >
                      {label(id)} <span>×</span>
                    </button>
                  ))}
                </div>
                <div
                  className="symptom-results"
                  aria-label="Available symptoms"
                >
                  {visibleSymptoms.length ? (
                    visibleSymptoms.map((s) => (
                      <button
                        disabled={busy}
                        aria-pressed={session.symptoms.includes(s.id)}
                        key={s.id}
                        onClick={() => toggle(s.id)}
                      >
                        <span>{s.label}</span>
                        <span>
                          {session.symptoms.includes(s.id) ? "−" : "+"}
                        </span>
                      </button>
                    ))
                  ) : (
                    <p>
                      No supported symptom matches. Try another word. Unlisted
                      symptoms cannot be included in this model.
                    </p>
                  )}
                </div>
                {!query.trim() && (
                  <button
                    className="browse-symptoms text-button"
                    onClick={() => setBrowseAll(!browseAll)}
                  >
                    {browseAll
                      ? "Show common symptoms"
                      : `Browse all ${registry.symptoms.length} symptoms`}
                  </button>
                )}
                <div className="followups">
                  <h3>A few follow-up questions</h3>
                  {registry.questions.map((q) => {
                    const a = session.followups.find(
                      (f) => f.question_id === q.id,
                    );
                    return (
                      <div className="question" key={q.id}>
                        <label htmlFor={q.id}>{q.prompt}</label>
                        <select
                          disabled={busy}
                          id={q.id}
                          value={a?.answer || ""}
                          onChange={(e) => followup(q.id, e.target.value)}
                        >
                          <option value="" disabled>
                            Choose an answer (optional)
                          </option>
                          {q.options.map((o) => (
                            <option key={o}>{o}</option>
                          ))}
                        </select>
                        {q.suggests &&
                          a?.answer === "Yes" &&
                          !session.symptoms.includes(q.suggests) && (
                            <label className="confirm-check">
                              <input
                                disabled={busy}
                                type="checkbox"
                                checked={a.confirmed}
                                onChange={(e) =>
                                  followup(q.id, a.answer, e.target.checked)
                                }
                              />
                              Confirm adding {label(q.suggests).toLowerCase()}{" "}
                              to model inputs
                            </label>
                          )}
                      </div>
                    );
                  })}
                  <small>
                    Duration is context only. Suggested symptoms need your
                    confirmation.
                  </small>
                </div>
                <button
                  className="primary"
                  disabled={selectedCount < 2 || busy}
                  onClick={assess}
                >
                  {busy ? "Reviewing your symptoms…" : "See possible matches"}{" "}
                  <span>↗</span>
                </button>
                {busy && (
                  <button
                    className="text-button"
                    onClick={() => controller.current?.abort()}
                  >
                    Cancel request ×
                  </button>
                )}
                <p className="fine-print">
                  Educational use only. If symptoms are severe, sudden or
                  worsening, seek medical care. This demo cannot assess urgency.
                </p>
              </div>
              <aside className="result-column">
                {session.result ? (
                  <Results
                    result={session.result}
                    fictional={!!session.sampleId}
                  />
                ) : (
                  <div className="empty-result">
                    <span className="icon-tile">
                      <Icon name="summary" />
                    </span>
                    <span className="eyebrow">Your results</span>
                    <h2>Your results will appear here.</h2>
                    <p>
                      Select symptoms, or open a fictional walkthrough. You’ll
                      see ranked matches, contributing features and the limits
                      of the model.
                    </p>
                  </div>
                )}
                {session.sampleId && explanation && (
                  <div className="explanation">
                    <span className="eyebrow">
                      {explanation.mode === "prepared"
                        ? "Prepared explanation"
                        : "Gemini · fictional case only"}
                    </span>
                    <p>{explanation.text}</p>
                    <small>{explanation.notice}</small>
                    <button
                      className="text-button"
                      disabled={explaining}
                      onClick={enhanceSample}
                    >
                      {explaining
                        ? "Opening the fictional explanation…"
                        : "Try the optional AI explanation ↗"}
                    </button>
                  </div>
                )}
                {session.result && (
                  <button
                    className="text-button"
                    onClick={() => changeTab("summary")}
                  >
                    Review your summary ↗
                  </button>
                )}
              </aside>
            </div>
          </>
        )}
        {tab === "reports" && (
          <div className="reports-workspace">
            <div className="report-intro">
              <span className="eyebrow">REPORT READER</span>
              <h2>Read and review a lab report</h2>
              <p>
                PDF, JPG or PNG. Up to three files, 10 MB each. PDFs can contain
                up to ten pages; images up to 20 megapixels.
              </p>
              <p className="muted">
                Files are sent directly to the Python backend for local text
                extraction and OCR. They are not sent to Gemini. Lab findings do
                not influence symptom matches.
              </p>
              <label className={`upload-zone ${extracting ? "disabled" : ""}`}>
                <input
                  ref={fileRef}
                  type="file"
                  accept=".pdf,.jpg,.jpeg,.png"
                  multiple
                  disabled={extracting || session.reports.length >= 3}
                  onChange={(e) => void processFiles(e.target.files)}
                />
                <span>↥</span>
                <strong>Choose reports</strong>
                <small>Processed one at a time</small>
              </label>
              {extracting && (
                <button
                  className="text-button"
                  onClick={() => extractionController.current?.abort()}
                >
                  Cancel extraction ×
                </button>
              )}
              <button
                disabled={extracting || session.reports.length >= 3}
                className="text-button"
                onClick={() =>
                  setSession((s) => ({
                    ...s,
                    reports: [
                      ...s.reports,
                      {
                        id: crypto.randomUUID(),
                        name: "Fictional lab report",
                        status: "done",
                        findings: fictionalFindings,
                        warnings: [
                          "Fictional example. Platelet reference range was intentionally left missing.",
                        ],
                      },
                    ],
                  }))
                }
              >
                Use a fictional report instead ↗
              </button>
            </div>
            <div className="report-results">
              {session.reports.length ? (
                session.reports.map((r) => (
                  <article className="report-card" key={r.id}>
                    <div className="section-heading">
                      <h3>{r.name}</h3>
                      <button
                        aria-label={`Remove ${r.name}`}
                        disabled={extracting}
                        onClick={() =>
                          setSession((s) => ({
                            ...s,
                            reports: s.reports.filter((x) => x.id !== r.id),
                          }))
                        }
                      >
                        ×
                      </button>
                    </div>
                    <p role="status" className="report-status">
                      {r.status === "extracting"
                        ? "Reading text and checking scanned pages…"
                        : r.status === "queued"
                          ? "Waiting in queue…"
                          : r.status === "done"
                            ? `${r.findings.length} findings · review against your original`
                            : r.status === "cancelled"
                              ? "Extraction cancelled. Select the file again to retry."
                              : "Could not extract this report."}
                    </p>
                    {r.status === "extracting" && (
                      <progress aria-label="Extraction in progress" />
                    )}
                    {r.error && <p className="message error">{r.error}</p>}
                    {r.warnings.map((w) => (
                      <p className="report-warning" key={w}>
                        {w}
                      </p>
                    ))}
                    {r.findings.map((f, i) => (
                      <div className="finding" key={i}>
                        <div className="finding-meta">
                          <input
                            className="test-name"
                            aria-label={`Test name for row ${i + 1} in ${r.name}`}
                            maxLength={80}
                            value={f.test_name}
                            onChange={(e) =>
                              editFinding(r, i, "test_name", e.target.value)
                            }
                          />
                          <small>
                            {f.method === "manual"
                              ? "Source page not recorded"
                              : `Page ${f.source_page}`}{" "}
                            · {f.method.replace("_", " ")}
                            {f.needs_review ? " · Review required" : ""}
                          </small>
                        </div>
                        <div className="finding-fields">
                          {(["value", "unit", "reference_range"] as const).map(
                            (key) => (
                              <label key={key}>
                                {key === "reference_range"
                                  ? "Reference range"
                                  : key === "value"
                                    ? "Value"
                                    : "Unit"}
                                <input
                                  aria-label={`${f.test_name} ${key.replace("_", " ")} in ${r.name}`}
                                  maxLength={80}
                                  value={f[key] || ""}
                                  placeholder="Not extracted"
                                  onChange={(e) =>
                                    editFinding(r, i, key, e.target.value)
                                  }
                                />
                              </label>
                            ),
                          )}
                        </div>
                        <label className="confirm-check">
                          <input
                            type="checkbox"
                            checked={f.needs_review}
                            onChange={(e) =>
                              updateReport(r.id, {
                                findings: r.findings.map((item, j) =>
                                  i === j
                                    ? {
                                        ...item,
                                        needs_review: e.target.checked,
                                      }
                                    : item,
                                ),
                              })
                            }
                          />
                          Needs review against the original
                        </label>
                        <button
                          className="text-button"
                          onClick={() =>
                            updateReport(r.id, {
                              findings: r.findings.filter((_, j) => j !== i),
                            })
                          }
                        >
                          Remove row ×
                        </button>
                      </div>
                    ))}
                    {r.status === "done" && (
                      <button
                        className="text-button"
                        onClick={() =>
                          updateReport(r.id, {
                            findings: [
                              ...r.findings,
                              {
                                test_name: "New test",
                                value: null,
                                unit: null,
                                reference_range: null,
                                source_page: 0,
                                method: "manual",
                                needs_review: true,
                              },
                            ],
                          })
                        }
                      >
                        Add a missing finding +
                      </button>
                    )}
                  </article>
                ))
              ) : (
                <div className="empty-result">
                  <span className="eyebrow">Nothing uploaded</span>
                  <h3>Start with a report or an example.</h3>
                  <p>
                    Extracted values appear here with their page and method.
                    Missing or uncertain fields remain visible for review.
                  </p>
                </div>
              )}
              <button
                className="text-button"
                onClick={() => changeTab("summary")}
              >
                Continue to summary ↗
              </button>
            </div>
          </div>
        )}
        {tab === "summary" && (
          <div className="summary-grid">
            <div>
              <span className="eyebrow">YOUR SUMMARY</span>
              <h2>Your session summary</h2>
              <p>
                Your inputs stay in memory until you leave or clear this
                session. Download a copy if you want to keep it.
              </p>
              <button
                className="primary"
                disabled={exporting}
                onClick={download}
              >
                {exporting ? "Preparing pages…" : "Download PDF summary"}{" "}
                <span>↓</span>
              </button>
              <div className="summary-block">
                <h3>Model inputs</h3>
                <p>
                  {(session.result?.symptoms || session.symptoms)
                    .map(label)
                    .join(", ") || "No symptoms selected."}
                </p>
                {session.followups.map((f) => (
                  <p key={f.question_id}>
                    {
                      registry.questions.find((q) => q.id === f.question_id)
                        ?.prompt
                    }{" "}
                    <strong>{f.answer}</strong>
                  </p>
                ))}
              </div>
              <div className="summary-block">
                <h3>Reports</h3>
                <p>
                  {session.reports.length
                    ? `${session.reports.length} report(s), ${session.reports.reduce((n, r) => n + r.findings.length, 0)} extracted rows.`
                    : "No reports in this session."}
                </p>
                {session.reports.map((r) => (
                  <p key={r.id}>
                    {r.name} · {r.status}
                    {r.findings.some((f) => f.needs_review)
                      ? " · Needs review"
                      : ""}
                  </p>
                ))}
                <Link to="?tab=reports">Review report fields ↗</Link>
              </div>
              <details className="bmi">
                <summary>Optional adult BMI calculator ↗</summary>
                <p>
                  A separate arithmetic tool. It is never included in the
                  disease model and is not a diagnosis or a complete measure of
                  health.
                </p>
                <div className="finding-fields">
                  <label>
                    Height (cm)
                    <input
                      inputMode="decimal"
                      type="number"
                      min="80"
                      max="250"
                      value={height}
                      onChange={(e) => setHeight(e.target.value)}
                    />
                  </label>
                  <label>
                    Weight (kg)
                    <input
                      inputMode="decimal"
                      type="number"
                      min="20"
                      max="400"
                      value={weight}
                      onChange={(e) => setWeight(e.target.value)}
                    />
                  </label>
                </div>
                <output>
                  {bmi
                    ? `BMI: ${bmi} kg/m²`
                    : "Enter adult height and weight within the displayed limits."}
                </output>
              </details>
            </div>
            <aside>
              {session.result ? (
                <Results
                  result={session.result}
                  fictional={!!session.sampleId}
                />
              ) : (
                <div className="empty-result">
                  <h3>No model result yet.</h3>
                  <p>
                    You can export the inputs and report findings without
                    running an assessment.
                  </p>
                  <Link to="?tab=symptoms">Explore symptoms ↗</Link>
                </div>
              )}
            </aside>
          </div>
        )}
      </section>
      <footer className="workspace-footer">
        <span>Educational exploration. No clinical validation.</span>
        <div>
          <Link to="/methods">Methodology</Link>
          <Link to="/privacy">Privacy</Link>
          <button onClick={() => setMotion(!motion)}>
            Motion {motion ? "on" : "off"}
          </button>
        </div>
      </footer>
    </main>
  );
}
export function Results({
  result,
  fictional,
}: {
  result: Assessment;
  fictional: boolean;
}) {
  return (
    <div className="results">
      <span className="eyebrow">
        {fictional
          ? "Fictional walkthrough · precomputed model output"
          : "Live model output"}
      </span>
      <h2>Possible matches</h2>
      <p className="muted">
        Ranked matches within a small dataset. These are not probabilities of
        having a condition.
      </p>
      {result.matches.map((m) => (
        <details className="match" key={m.condition} open={m.rank === 1}>
          <summary>
            <small>0{m.rank}</small>
            <strong>{m.label}</strong>
            <span>+</span>
          </summary>
          <div className="match-body">
            <p>{m.description}</p>
            <h4>Supporting selections</h4>
            <p>
              {m.supporting_symptoms.map(label).join(", ") ||
                "No positive selected contribution among these inputs."}
            </p>
            <details className="explanation-details">
              <summary>Why these symptoms matched</summary>
              <div className="mini-contributions">
                {m.contributions.map((c) => (
                  <div key={c.symptom_id}>
                    <span>
                      {c.label}{" "}
                      <small>{c.present ? "selected" : "not selected"}</small>
                    </span>
                    <b className={c.contribution >= 0 ? "positive" : ""}>
                      {c.contribution >= 0 ? "+" : ""}
                      {c.contribution.toFixed(3)}
                    </b>
                  </div>
                ))}
              </div>
              <p className="fine-print">
                Largest SHAP contributions for this class, in internal
                model-score units. Positive raises the score; negative lowers
                it. Absence can also contribute. These values explain the model,
                not a medical cause.
              </p>
            </details>
            <h4>Learn about this condition</h4>
            {m.sources.map((s) => (
              <a
                className="source-link"
                key={s.id}
                href={s.url}
                target="_blank"
                rel="noreferrer"
              >
                {s.title} ↗<small>{s.publisher}</small>
              </a>
            ))}
          </div>
        </details>
      ))}
      <details className="model-limits">
        <summary>What this result cannot tell you</summary>
        <ul>
          {result.limitations.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
        <small>{result.model_version}</small>
      </details>
      <p className="fine-print">{result.notice}</p>
    </div>
  );
}
