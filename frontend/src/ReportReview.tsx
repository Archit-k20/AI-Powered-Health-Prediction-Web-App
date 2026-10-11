import { lazy, Suspense, useEffect, useState } from "react";
const PdfPreview = lazy(() =>
  import("./PdfPreview").catch(() => ({
    default: () => (
      <p role="status">Preview unavailable. Open the original report below.</p>
    ),
  })),
);
import type { Report } from "./state";
import type { Evidence, Finding } from "./types";

const kindLabel: Record<string, string> = {
  laboratory: "Laboratory report",
  clinical_note: "Clinical note",
  discharge_summary: "Discharge summary",
  radiology: "Written radiology report",
  template: "Unfilled report template",
  unknown: "Report type uncertain",
};
export function ReportReview({
  report: r,
  update,
  remove,
  confirm,
  conflicts,
  disabled,
  continueToSymptoms,
}: {
  report: Report;
  update: (patch: Partial<Report>) => void;
  remove: () => void;
  confirm: (id: string, checked: boolean) => void;
  conflicts: Set<string>;
  disabled: boolean;
  continueToSymptoms: () => void;
}) {
  const [page, setPage] = useState(1);
  const [url, setUrl] = useState("");
  const [quote, setQuote] = useState("");
  const [highlight, setHighlight] = useState<number[] | undefined>();
  useEffect(() => {
    if (!r.file || !URL.createObjectURL) return;
    const source = URL.createObjectURL(
      new Blob([r.file], {
        type: /\.pdf$/i.test(r.name) ? "application/pdf" : r.file.type,
      }),
    );
    setUrl(source);
    return () => URL.revokeObjectURL(source);
  }, [r.file, r.name]);
  function source(e?: Evidence | null) {
    if (!e) return;
    setPage(e.page);
    setQuote(e.quote);
    setHighlight(e.bbox || undefined);
  }
  const pageDetail = r.pageDetails?.find((p) => p.page === page);
  const mentions = r.mentions || [];
  const candidates = mentions.filter((m) => m.category === "symptom");
  const context = mentions.filter((m) => m.category !== "symptom");
  const kinds = [
    ...new Set(r.pageDetails?.map((p) => kindLabel[p.document_type]) || []),
  ];
  function edit(index: number, key: keyof Finding, value: string) {
    update({
      findings: r.findings.map((f, i) =>
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
  return (
    <article className="report-card">
      <div className="section-heading">
        <h3>{r.name}</h3>
        <button
          aria-label={`Remove ${r.name}`}
          disabled={disabled}
          onClick={remove}
        >
          ×
        </button>
      </div>
      <p role="status" className="report-status">
        {r.status === "extracting"
          ? "Reading pages, checking layout and identifying mentions…"
          : r.status === "queued"
            ? "Waiting in queue…"
            : r.status === "done"
              ? `${r.findings.length} lab findings · ${mentions.length} text mentions · review against your original`
              : r.status === "cancelled"
                ? "Extraction cancelled. Select the file again to retry."
                : "Could not extract this report."}
      </p>
      {r.status === "extracting" && (
        <progress aria-label="Extraction in progress" />
      )}
      {r.error && <p className="message error">{r.error}</p>}
      {r.status === "done" && (
        <>
          <div className="report-kind">
            {kinds.map((k) => (
              <span key={k}>{k}</span>
            ))}
            <span>
              {r.partial
                ? "Review needed"
                : "Reading complete · verify findings"}
            </span>
          </div>
          {r.warnings.length > 0 && (
            <details
              className="report-warnings"
              open={r.findings.length + mentions.length === 0}
            >
              <summary>
                {r.warnings.length} reading{" "}
                {r.warnings.length === 1 ? "note" : "notes"}
              </summary>
              {r.warnings.map((w, i) => (
                <p className="report-warning" key={i}>
                  {w}
                </p>
              ))}
            </details>
          )}
          <div className="report-review-grid">
            <aside
              className="report-original"
              aria-label={`Original and evidence for ${r.name}`}
            >
              <h4>1. Check the original</h4>
              {r.pageDetails && r.pageDetails.length > 1 && (
                <label>
                  Source page
                  <select
                    aria-label={`Source page in ${r.name}`}
                    value={page}
                    onChange={(e) => {
                      setPage(+e.target.value);
                      setQuote("");
                      setHighlight(undefined);
                    }}
                  >
                    {r.pageDetails.map((p) => (
                      <option key={p.page} value={p.page}>
                        Page {p.page} · {kindLabel[p.document_type]}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              {url &&
                (/\.pdf$/i.test(r.name) ? (
                  <Suspense
                    fallback={<p role="status">Loading document preview…</p>}
                  >
                    <PdfPreview
                      file={r.file!}
                      page={page}
                      highlight={highlight}
                    />
                  </Suspense>
                ) : (
                  <img src={url} alt={`Original uploaded report ${r.name}`} />
                ))}
              {url && (
                <a
                  href={`${url}#page=${page}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  Open original report ↗
                </a>
              )}
              {quote && (
                <div className="source-quote" role="status">
                  <small>Selected evidence · page {page}</small>
                  <blockquote>{quote}</blockquote>
                </div>
              )}
              {pageDetail && (
                <details className="reading-transcript" open={!url}>
                  <summary>Read extracted text · page {page}</summary>
                  <p className="muted">
                    {!r.file
                      ? "Prepared fictional transcript."
                      : pageDetail.method === "ocr"
                        ? "OCR transcript — compare with the image; it can contain errors."
                        : "Text read from the PDF."}
                  </p>
                  <pre>
                    {pageDetail.text || "No readable text on this page."}
                  </pre>
                </details>
              )}
              {!url && !pageDetail && (
                <p className="muted">
                  This fictional example has no uploaded original.
                </p>
              )}
            </aside>
            <div className="report-structured">
              <section className="report-section">
                <h4>2. Confirm current symptoms</h4>
                <p className="muted">
                  A report may be old or describe someone else. Nothing is
                  selected automatically.
                </p>
                {!candidates.length && (
                  <div className="report-empty">
                    <strong>No symptom inputs identified</strong>
                    <p>
                      This report alone cannot supply the symptom model. You can
                      still review its findings and tell us about symptoms
                      yourself.
                    </p>
                  </div>
                )}
                {candidates.map((m) => {
                  const conflict =
                    !!m.symptom_id && conflicts.has(m.symptom_id);
                  const eligible = m.eligible && !conflict;
                  return (
                    <div className="mention" key={m.id}>
                      <strong>{m.label}</strong>
                      <p className="mention-status">
                        {
                          {
                            affirmed: "Mentioned",
                            negated: "Reported as absent",
                            uncertain: "Uncertain",
                          }[m.polarity]
                        }{" "}
                        ·{" "}
                        {m.timing === "unspecified"
                          ? "timing needs confirmation"
                          : m.timing}{" "}
                        · {m.subject}
                      </p>
                      <blockquote>{m.evidence.quote}</blockquote>
                      <button
                        className="text-button"
                        onClick={() => source(m.evidence)}
                      >
                        View source · page {m.evidence.page}
                      </button>
                      <p className="muted">
                        {conflict
                          ? "Other mentions disagree or refer to a different time/person. Clarify this manually in Symptoms."
                          : m.review_reason}
                      </p>
                      {eligible && (
                        <label className="confirm-check">
                          <input
                            type="checkbox"
                            checked={
                              r.confirmedMentions?.includes(m.id) || false
                            }
                            onChange={(e) => confirm(m.id, e.target.checked)}
                          />
                          I currently have {m.label.toLowerCase()}
                        </label>
                      )}
                      {!eligible && (
                        <span className="context-only">
                          Context only · not a model input
                        </span>
                      )}
                    </div>
                  );
                })}
                <button className="primary" onClick={continueToSymptoms}>
                  Review symptoms & continue <span>→</span>
                </button>
              </section>
              <section className="report-section">
                <h4>3. Review lab values</h4>
                <p className="muted">
                  Copied values and ranges only. They do not change symptom
                  predictions.
                </p>
                {!r.findings.length && <p>No lab values identified.</p>}
                {r.findings.map((f, i) => (
                  <div className="finding" key={i}>
                    <div className="finding-meta">
                      <input
                        className="test-name"
                        aria-label={`Test name for row ${i + 1} in ${r.name}`}
                        maxLength={80}
                        value={f.test_name}
                        onChange={(e) => edit(i, "test_name", e.target.value)}
                      />
                      <small>
                        {f.method === "manual"
                          ? "Source page not recorded"
                          : `Page ${f.source_page}`}{" "}
                        · {f.method.replace("_", " ")}
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
                              onChange={(e) => edit(i, key, e.target.value)}
                            />
                          </label>
                        ),
                      )}
                    </div>
                    {f.evidence && (
                      <details className="finding-evidence">
                        <summary>Source text · page {f.evidence.page}</summary>
                        <blockquote>{f.evidence.quote}</blockquote>
                        <button
                          className="text-button"
                          onClick={() => source(f.evidence)}
                        >
                          View source page
                        </button>
                      </details>
                    )}
                    {f.review_reason && (
                      <p className="muted">{f.review_reason}</p>
                    )}
                    <label className="confirm-check">
                      <input
                        type="checkbox"
                        checked={f.needs_review}
                        onChange={(e) =>
                          update({
                            findings: r.findings.map((item, j) =>
                              j === i
                                ? { ...item, needs_review: e.target.checked }
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
                        update({
                          findings: r.findings.filter((_, j) => j !== i),
                        })
                      }
                    >
                      Remove row ×
                    </button>
                  </div>
                ))}
                <button
                  className="text-button"
                  onClick={() =>
                    update({
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
              </section>
              {!!context.length && (
                <section className="report-section">
                  <h4>Other document context</h4>
                  <p className="muted">
                    Claims from the document, not conclusions made by this app.
                    These do not become model inputs.
                  </p>
                  {context.map((m) => (
                    <div className="mention" key={m.id}>
                      <small>
                        {m.category} · {m.polarity} · {m.timing} · {m.subject}
                      </small>
                      <p>{m.label}</p>
                      <button
                        className="text-button"
                        onClick={() => source(m.evidence)}
                      >
                        View source · page {m.evidence.page}
                      </button>
                    </div>
                  ))}
                </section>
              )}
            </div>
          </div>
        </>
      )}
    </article>
  );
}
