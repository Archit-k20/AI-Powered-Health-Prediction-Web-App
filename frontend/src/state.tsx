import {
  createContext,
  useContext,
  useState,
  useEffect,
  type ReactNode,
} from "react";
import type { Assessment, Finding, Followup } from "./types";
export type Report = {
  id: string;
  name: string;
  status: "queued" | "extracting" | "done" | "error" | "cancelled";
  findings: Finding[];
  warnings: string[];
  error?: string;
};
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
