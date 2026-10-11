import { jsPDF } from "jspdf";
import type { Session } from "./state";
import { selectedSymptoms, confirmedReportSymptoms } from "./state";
import registry from "../../shared/symptoms.json";
export function createSummary(session: Session) {
  const doc = new jsPDF();
  let y = 23;
  let page = 1;
  const clean = (value: string) =>
    value
      .replace(/[µμ]/g, "u")
      .replace(/×/g, "x")
      .replace(/³/g, "^3")
      .replace(/²/g, "^2")
      .replace(/⁹/g, "^9")
      .replace(/≥/g, ">=")
      .replace(/≤/g, "<=")
      .replace(/±/g, "+/-")
      .normalize("NFKD")
      .replace(/[–—]/g, "-")
      .replace(/→|↗/g, "->")
      .replace(/[^\x20-\x7E\n]/g, "");
  function footer() {
    doc.setFontSize(8);
    doc.setTextColor(90);
    doc.text(
      "The Shape of Health | Educational only. Not clinically validated.",
      18,
      285,
    );
    doc.text(String(page), 190, 285);
    doc.setTextColor(30);
  }
  function line(text: string, size = 11) {
    doc.setFontSize(size);
    const lines: string[] = doc.splitTextToSize(clean(text), 174);
    for (const item of lines) {
      if (y > 271) {
        footer();
        doc.addPage();
        page++;
        y = 23;
      }
      doc.text(item, 18, y);
      y += size * 0.48 + 1.5;
    }
    y += 2;
  }
  function title(text: string) {
    y += 5;
    line(text, 16);
  }
  line("The Shape of Health", 25);
  line(
    session.sampleId
      ? `Fictional walkthrough: ${session.sampleId}`
      : "Session summary",
    13,
  );
  line(
    "Educational exploration only. This is not a diagnosis, a risk assessment or a substitute for professional medical care. For severe or sudden symptoms, seek urgent medical help.",
  );
  title("Selected model inputs");
  const ids = session.result?.symptoms || selectedSymptoms(session);
  line(
    ids.length
      ? ids
          .map((id) => registry.symptoms.find((s) => s.id === id)?.label || id)
          .join(", ")
      : "No symptoms selected.",
  );
  title("Context");
  const context =
    session.result?.context ||
    session.followups.map(
      (f) =>
        `${registry.questions.find((q) => q.id === f.question_id)?.prompt || f.question_id} ${f.answer}${f.confirmed ? " (symptom addition confirmed)" : ""}`,
    );
  context.forEach((x) => line(x));
  if (!context.length) line("No follow-up answers recorded.");
  if (session.result) {
    title("Ranked model matches");
    line(
      `Model: ${session.result.model_version}. Rankings are dataset matches, not individual disease probabilities.`,
    );
    for (const match of session.result.matches) {
      title(`${match.rank}. ${match.label}`);
      line(match.description);
      line(
        `Supporting selected symptoms: ${match.supporting_symptoms.map((id) => registry.symptoms.find((s) => s.id === id)?.label || id).join(", ") || "None with a positive contribution."}`,
      );
      line(
        "Largest SHAP contributions for this class (internal model-score units):",
      );
      match.contributions.forEach((c) =>
        line(
          `${c.label}: ${c.contribution >= 0 ? "+" : ""}${c.contribution.toFixed(4)}; ${c.present ? "selected" : "not selected"}`,
          10,
        ),
      );
      line(
        "Only the largest contributions are listed; they do not sum to the full model output. Contributions describe model behavior, not medical causes.",
        10,
      );
      match.sources.forEach((s) => {
        line(`${s.title} | ${s.publisher}`, 10);
        line(s.url, 9);
      });
    }
    title("Model limitations");
    session.result.limitations.forEach((x) => line(x));
  } else {
    title("Model results");
    line(
      "No completed assessment. No prediction is inferred from this summary.",
    );
  }
  title("Report findings");
  line(
    "Report values are kept separate from the symptom model. Verify all values against the original report. Reference ranges are copied, not medical interpretations.",
  );
  if (!session.reports.length) line("No reports in this session.");
  for (const report of session.reports) {
    title(report.name);
    line(`Status: ${report.status}`);
    if (report.error) line(report.error);
    for (const page of report.pageDetails || [])
      line(
        `Page ${page.page}: ${page.document_type.replaceAll("_", " ")}; ${page.method}; ${page.quality}.`,
        10,
      );
    for (const mention of report.mentions || []) {
      const included =
        mention.eligible &&
        report.confirmedMentions?.includes(mention.id) &&
        !!mention.symptom_id &&
        confirmedReportSymptoms(session.reports).includes(mention.symptom_id);
      line(
        `${mention.label} [${mention.category}; ${mention.polarity}; ${mention.timing}; ${mention.subject}]. ${included ? "Confirmed as a current model input." : "Document context only; not a model input."}`,
        10,
      );
      line(
        `Source page ${mention.evidence.page} (${mention.evidence.method}): ${mention.evidence.quote}`,
        10,
      );
    }
    for (const f of report.findings) {
      line(
        `${f.test_name}: ${f.value || "not extracted"} ${f.unit || ""}; reference: ${f.reference_range || "not extracted"}; ${f.method === "manual" ? "source page not recorded" : `page ${f.source_page}`}; ${f.method}; review ${f.needs_review ? "required" : "still recommended"}.`,
        10,
      );
      if (f.evidence) line(`Source: ${f.evidence.quote}`, 9);
    }
    report.warnings.forEach((w) => line(w, 10));
  }
  title("Privacy and provenance");
  line(
    "This download was generated in your browser. Health inputs are held in application memory. Live inputs and reports are processed by the configured Python backend; they are never submitted to Gemini. The optional generative example endpoint accepts predefined fictional case IDs only. You control where you save or share this PDF.",
  );
  footer();
  return doc;
}
