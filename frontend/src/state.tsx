import {
  createContext,
  useContext,
  useState,
  useEffect,
  type ReactNode,
} from "react";
import type {
  Assessment,
  Finding,
  Followup,
  ReportMention,
  ReportPage,
} from "./types";
export type Report = {
  id: string;
  name: string;
  status: "queued" | "extracting" | "done" | "error" | "cancelled";
  findings: Finding[];
  warnings: string[];
  mentions?: ReportMention[];
  pageDetails?: ReportPage[];
  confirmedMentions?: string[];
  file?: File;
  partial?: boolean;
  error?: string;
};
export function conflictingReportSymptoms(reports: Report[]) {
  const mentions = reports.flatMap((r) => r.mentions || []);
  return new Set(
    mentions
      .filter(
        (m) =>
          m.symptom_id &&
          mentions.some(
            (other) =>
              other.symptom_id === m.symptom_id &&
              other.polarity === "affirmed" &&
              other.timing === "unspecified" &&
              other.subject === "patient",
          ) &&
          (m.polarity !== "affirmed" ||
            m.timing !== "unspecified" ||
            m.subject !== "patient"),
      )
      .map((m) => m.symptom_id!),
  );
}
export function confirmedReportSymptoms(reports: Report[]) {
  const conflicts = conflictingReportSymptoms(reports);
  return [
    ...new Set(
      reports.flatMap((r) =>
        (r.mentions || [])
          .filter(
            (m) =>
              m.eligible &&
              m.symptom_id &&
              !conflicts.has(m.symptom_id) &&
              r.confirmedMentions?.includes(m.id),
          )
          .map((m) => m.symptom_id!),
      ),
    ),
  ];
}
export function selectedSymptoms(session: Session) {
  return [
    ...new Set([
      ...session.symptoms,
      ...confirmedReportSymptoms(session.reports),
    ]),
  ];
}
export type Session = {
  symptoms: string[];
  followups: Followup[];
  result: Assessment | null;
  reports: Report[];
  sampleId: string | null;
};
export const emptySession: Session = {
  symptoms: [],
  followups: [],
  result: null,
  reports: [],
  sampleId: null,
};
const SessionContext = createContext<{
  session: Session;
  setSession: React.Dispatch<React.SetStateAction<Session>>;
}>(null!);
const PreferencesContext = createContext<{
  motion: boolean;
  setMotion: (x: boolean) => void;
}>(null!);
export const useSession = () => useContext(SessionContext);
export const usePreferences = () => useContext(PreferencesContext);
export function Providers({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session>(emptySession);
  const [motion, setMotionState] = useState(() => {
    try {
      return (
        localStorage.getItem("health-motion") !== "off" &&
        !matchMedia("(prefers-reduced-motion: reduce)").matches
      );
    } catch {
      return false;
    }
  });
  useEffect(() => {
    const media = matchMedia("(prefers-reduced-motion: reduce)");
    const change = () => {
      if (media.matches) setMotionState(false);
    };
    media.addEventListener("change", change);
    return () => media.removeEventListener("change", change);
  }, []);
  useEffect(() => {
    document.documentElement.dataset.motion = motion ? "on" : "off";
  }, [motion]);
  function setMotion(value: boolean) {
    setMotionState(value);
    try {
      localStorage.setItem("health-motion", value ? "on" : "off");
    } catch {
      /* private mode */
    }
  }
  return (
    <PreferencesContext.Provider value={{ motion, setMotion }}>
      <SessionContext.Provider value={{ session, setSession }}>
        {children}
      </SessionContext.Provider>
    </PreferencesContext.Provider>
  );
}
