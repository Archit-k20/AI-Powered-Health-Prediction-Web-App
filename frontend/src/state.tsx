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
  sound: boolean;
  setSound: (x: boolean) => void;
  ping: () => void;
}>(null!);
export const useSession = () => useContext(SessionContext);
export const usePreferences = () => useContext(PreferencesContext);
let audio: AudioContext | undefined;
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
  const [sound, setSoundState] = useState(false);
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
  function setSound(value: boolean) {
    setSoundState(value);
    if (value) {
      audio ??= new AudioContext();
      void audio.resume();
    } else {
      void audio?.suspend();
    }
  }
  function ping() {
    if (!sound || !audio || document.hidden) return;
    const osc = audio.createOscillator();
    const gain = audio.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(220, audio.currentTime);
    osc.frequency.exponentialRampToValueAtTime(330, audio.currentTime + 0.18);
    gain.gain.setValueAtTime(0.022, audio.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audio.currentTime + 0.3);
    osc.connect(gain);
    gain.connect(audio.destination);
    osc.start();
    osc.stop(audio.currentTime + 0.31);
  }
  return (
    <PreferencesContext.Provider
      value={{ motion, setMotion, sound, setSound, ping }}
    >
      <SessionContext.Provider value={{ session, setSession }}>
        {children}
      </SessionContext.Provider>
    </PreferencesContext.Provider>
  );
}
