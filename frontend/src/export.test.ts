import { describe, it, expect } from "vitest";
import { createSummary } from "./export";
import { emptySession } from "./state";
import samples from "../../shared/samples.json";
import type { Assessment } from "./types";
describe("PDF export", () => {
  it("creates a paginated PDF with sources and educational notice", () => {
    const doc = createSummary({
      ...emptySession,
      sampleId: "cold",
      symptoms: samples[0].symptoms,
      result: samples[0].assessment as Assessment,
    });
    expect(doc.getNumberOfPages()).toBeGreaterThan(1);
    const output = doc.output();
    expect(output.startsWith("%PDF-")).toBe(true);
    expect(output).toContain("Educational");
    expect(output).toContain("medlineplus");
    expect(output).toContain("Selected model inputs");
  });
});
it("preserves the meaning of scientific units in the PDF font", () => {
  const doc = createSummary({
    ...emptySession,
    reports: [
      {
        id: "x",
        name: "Fictional units",
        status: "done",
        warnings: [],
        findings: [
          {
            test_name: "Platelets",
            value: "250",
            unit: "×10³/µL",
            reference_range: null,
            source_page: 1,
            method: "fictional",
            needs_review: true,
          },
        ],
      },
    ],
  });
  expect(doc.output()).toContain("x10^3/uL");
});

it("exports source evidence and separates confirmed symptoms from document claims", async () => {
  const { default: example } = await import("../../shared/report-example.json");
  const doc = createSummary({
    ...emptySession,
    reports: [
      {
        id: "note",
        name: "Fictional note",
        status: "done",
        warnings: [],
        findings: [],
        mentions: example.mentions as import("./types").ReportMention[],
        confirmedMentions: example.mentions
          .filter((m) => m.eligible)
          .map((m) => m.id),
      },
    ],
  });
  expect(doc.output()).toContain("Confirmed as a current model input.");
  expect(doc.output()).toContain("Document context only; not a model input.");
  expect(doc.output()).toContain("Patient denies shortness of breath.");
  expect(doc.getNumberOfPages()).toBeGreaterThan(1);
});
