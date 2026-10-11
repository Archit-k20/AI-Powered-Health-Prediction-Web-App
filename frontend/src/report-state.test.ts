import { describe, it, expect } from "vitest";
import {
  confirmedReportSymptoms,
  selectedSymptoms,
  emptySession,
  type Report,
} from "./state";
import example from "../../shared/report-example.json";
import type { ReportMention } from "./types";
const report = (): Report => ({
  id: "note",
  name: "Fictional note",
  status: "done",
  warnings: [],
  findings: [],
  mentions: example.mentions as ReportMention[],
  confirmedMentions: example.mentions.map((m) => m.id),
});
describe("report-derived model inputs", () => {
  it("ignores confirmations on negated, historical and hypothetical mentions", () => {
    expect(confirmedReportSymptoms([report()]).sort()).toEqual([
      "cough",
      "fatigue",
    ]);
  });
  it("blocks conflicting mentions across separate uploaded reports", () => {
    const a = report();
    const b: Report = {
      ...report(),
      id: "second",
      mentions: [
        {
          ...a.mentions!.find((m) => m.symptom_id === "cough")!,
          polarity: "negated",
          eligible: false,
        },
      ],
    };
    expect(confirmedReportSymptoms([a, b])).toEqual(["fatigue"]);
  });
  it("deduplicates confirmed and manually selected symptoms", () => {
    expect(
      selectedSymptoms({
        ...emptySession,
        symptoms: ["cough"],
        reports: [report()],
      }).sort(),
    ).toEqual(["cough", "fatigue"]);
  });
});
