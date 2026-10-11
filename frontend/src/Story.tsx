import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { usePreferences, useSession } from "./state";
import registry from "../../shared/symptoms.json";
import samples from "../../shared/samples.json";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
const Sculpture = lazy(() => import("./Sculpture"));
gsap.registerPlugin(ScrollTrigger);
const chapters = [
  "Signals",
  "Symptoms",
  "Patterns",
  "Context",
  "Reports",
  "Reasoning",
  "Limits",
  "Explore",
];
const field = [
  "headache",
  "fatigue",
  "high_fever",
  "cough",
  "itching",
  "nausea",
  "skin_rash",
  "breathlessness",
  "joint_pain",
  "runny_nose",
  "chills",
  "dizziness",
];
const label = (id: string) =>
  registry.symptoms.find((s) => s.id === id)?.label || id;
function ChapterMeta({ index, aside }: { index: number; aside: string }) {
  return (
    <div className="chapter-meta">
      <span>
        {String(index + 1).padStart(2, "0")} <span className="muted">/ 08</span>{" "}
        <b>{chapters[index]}</b>
      </span>
      <span>{aside}</span>
    </div>
  );
}
function ChapterNote({
  number,
  title,
  children,
}: {
  number: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="chapter-note">
      <div>
        <span className="eyebrow">{number} — A closer look</span>
        <h3>{title}</h3>
        <Link to="/methods">Read the method ↗</Link>
      </div>
      <p>{children}</p>
    </div>
  );
}
export default function Story() {
  const { motion, setMotion, sound, setSound, ping } = usePreferences();
  const { session, setSession } = useSession();
  const navigate = useNavigate();
  const [active, setActive] = useState(0);
  const [menu, setMenu] = useState(false);
  const [shape, setShape] = useState(1);
  const [context, setContext] = useState(0);
  const [report, setReport] = useState(false);
  const [depth, setDepth] = useState(40);
  const root = useRef<HTMLElement>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  const menuButton = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting)
            setActive(Number((e.target as HTMLElement).dataset.chapter));
        });
      },
      { rootMargin: "-25% 0px -50% 0px" },
    );
    root.current
      ?.querySelectorAll("[data-chapter]")
      .forEach((e) => observer.observe(e));
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!motion) return;
    const ctx = gsap.context(() => {
      gsap.utils
        .toArray<HTMLElement>(".reveal-line")
        .forEach((el) =>
          gsap.fromTo(
            el,
            { opacity: 0.18, y: 35 },
            {
              opacity: 1,
              y: 0,
              scrollTrigger: {
                trigger: el,
                start: "top 90%",
                end: "top 42%",
                scrub: true,
              },
            },
          ),
        );
      gsap.utils.toArray<HTMLElement>(".pin-scene").forEach((el) => {
        ScrollTrigger.create({
          trigger: el,
          start: "top top",
          end: "+=35%",
          pin: true,
          pinSpacing: true,
        });
      });
      gsap.utils
        .toArray<HTMLElement>(".parallax-word")
        .forEach((el) =>
          gsap.to(el, {
            xPercent: -7,
            scrollTrigger: {
              trigger: el,
              start: "top bottom",
              end: "bottom top",
              scrub: 1,
            },
          }),
        );
    }, root);
    return () => ctx.revert();
  }, [motion]);
  useEffect(() => {
    if (menu) dialog.current?.showModal();
    else if (dialog.current?.open) {
      dialog.current.close();
      menuButton.current?.focus();
    }
  }, [menu]);
  function jump(i: number) {
    setMenu(false);
    document
      .getElementById(chapters[i].toLowerCase())
      ?.scrollIntoView({ behavior: motion ? "smooth" : "instant" });
    ping();
  }
  function select(id: string) {
    setSession((s) => ({
      ...s,
      sampleId: null,
      result: null,
      symptoms: s.symptoms.includes(id)
        ? s.symptoms.filter((x) => x !== id)
        : [...s.symptoms, id],
    }));
    ping();
  }
  return (
    <main id="main" ref={root} className="story">
      <section id="signals" data-chapter="0" className="hero">
        <div className="hero-intro">
          <p>
            An exploration of symptoms,
            <br />
            patterns, and possibility.
          </p>
          <p>
            A college idea, reimagined.
            <br />
            An open experiment.
          </p>
        </div>
        <div className="hero-composition">
          <h1>
            <span>
              The Shape <sup>of</sup>
            </span>
            <span className="health-word">Health</span>
          </h1>
          <Suspense
            fallback={
              <div className="sculpture-loading" aria-hidden="true">
                ∿
              </div>
            }
          >
            {active === 0 ? (
              <Sculpture shape={shape} onShape={setShape} />
            ) : (
              <div className="sculpture-placeholder" />
            )}
          </Suspense>
          <span className="drag-hint">Drag to reshape ↔</span>
        </div>
        <div className="hero-caption">
          <p>
            Our bodies send signals.
            <br />
            Models look for patterns.
            <br />
            <span className="muted">Understanding takes more.</span>
          </p>
          <button className="begin" onClick={() => jump(1)}>
            <span className="circle-arrow">↘</span>Begin the story
          </button>
        </div>
        <div className="shape-control">
          <label htmlFor="shape">
            Change the shape of a signal <span>↔</span>
          </label>
          <div className="shape-labels">
            {["Signals", "Patterns", "Perspective"].map((s, i) => (
              <button
                key={s}
                className={Math.round(shape) === i ? "selected" : ""}
                onClick={() => {
                  setShape(i);
                  ping();
                }}
              >
                <small>0{i + 1}</small> {s}
              </button>
            ))}
          </div>
          <input
            id="shape"
            aria-label="Reshape the sculpture"
            type="range"
            min="0"
            max="2"
            step="0.01"
            value={shape}
            onChange={(e) => setShape(+e.target.value)}
          />
          <small>
            {shape < 0.7
              ? "One signal. Many possible meanings."
              : shape < 1.4
                ? "A structure shaped by the examples it has seen."
                : "A different perspective changes what we notice."}
          </small>
        </div>
      </section>
      <section className="introduction">
        <span className="eyebrow">
          An interactive exhibition
          <br />
          by Archit
        </span>
        <div>
          <p className="reveal-line">
            First, we notice
            <br />a signal.
          </p>
          <p className="reveal-line">
            Then, we look
            <br />
            for a pattern.
          </p>
          <p className="reveal-line">
            And a better
            <br />
            question emerges.
          </p>
          <p className="intro-end">
            Eight chapters. One experiment in understanding.
            <br />
            Explore the story, or <Link to="/workspace">try the tools ↗</Link>
          </p>
        </div>
      </section>
      <section id="symptoms" data-chapter="1" className="chapter dark">
        <div className="chapter-inner">
          <ChapterMeta index={1} aside="What we notice" />
          <h2>
            Something
            <br />
            feels different.
          </h2>
          <div className="symptom-field">
            {field.map((id, i) => (
              <button
                key={id}
                aria-pressed={session.symptoms.includes(id)}
                style={{ "--delay": `${i * 0.21}s` } as React.CSSProperties}
                onClick={() => select(id)}
              >
                <span>+</span>
                {label(id)}
              </button>
            ))}
          </div>
          <div className="scene-caption">
            <p>
              Choose a few signals.
              <br />
              <span className="muted">
                There is no conclusion in a single word.
              </span>
            </p>
            <Link to="/workspace">
              Explore{" "}
              {session.symptoms.length
                ? `${session.symptoms.length} selections`
                : "symptoms"}{" "}
              ↗
            </Link>
          </div>
          <ChapterNote number="02" title="Language is the first input.">
            A symptom has many names. This project brings everyday words and a
            fixed dataset vocabulary together. Your selections are explicit and
            editable. Two people can describe the same experience differently; a
            small vocabulary cannot capture everything.
          </ChapterNote>
        </div>
      </section>
      <section id="patterns" data-chapter="2" className="chapter">
        <div className="chapter-inner">
          <ChapterMeta index={2} aside="From words to a pattern" />
          <h2 className="parallax-word">The pattern.</h2>
          <div className="pattern-scene">
            <div className="input-words">
              {(session.symptoms.length ? session.symptoms : field.slice(0, 3))
                .slice(0, 5)
                .map((s) => (
                  <span key={s}>{label(s)}</span>
                ))}
            </div>
            <span className="flow-arrow">→</span>
            <div
              className="binary-field"
              aria-label="Illustration of binary symptom encoding"
            >
              {Array.from({ length: 130 }, (_, i) => (
                <span className={i % 13 < 3 ? "on" : ""} key={i}>
                  {i % 13 < 3 ? "1" : "0"}
                </span>
              ))}
            </div>
            <span className="flow-arrow">→</span>
            <div className="model-form">
              <span>100</span>
              <small>decision trees</small>
            </div>
          </div>
          <div className="scene-caption">
            <p>
              Selected becomes one.
              <br />
              Unselected becomes zero.
            </p>
            <span className="muted">
              An illustration of the encoding, not a live prediction.
            </span>
          </div>
          <ChapterNote
            number="03"
            title="A model learns the world it is given."
          >
            The random forest compares combinations of{" "}
            {registry.symptoms.length} recorded features. Its training set
            contains 243 unique patterns, across 41 labels. That is a tiny view
            of health. A match reflects these examples, not the full range of
            possible causes.
          </ChapterNote>
        </div>
      </section>
      <section id="context" data-chapter="3" className="chapter dark">
        <div className="chapter-inner">
          <ChapterMeta index={3} aside="The question after the question" />
          <h2>
            Context changes
            <br />
            the conversation.
          </h2>
          <div className="context-scene">
            <div className="context-orbit" aria-hidden="true">
              {Array.from({ length: 14 }, (_, i) => (
                <i
                  key={i}
                  style={{ transform: `rotate(${i * 13 + context * 12}deg)` }}
                />
              ))}
            </div>
            <p className="context-word">
              {["a headache", "since this morning", "with a fever"][context]}
            </p>
            <div className="context-options">
              {["The signal", "The duration", "Another signal"].map((s, i) => (
                <button
                  key={s}
                  aria-pressed={context === i}
                  onClick={() => {
                    setContext(i);
                    ping();
                  }}
                >
                  <small>0{i + 1}</small>
                  {s} ↗
                </button>
              ))}
            </div>
          </div>
          <p className="context-explanation">
            {
              [
                "Start with what was actually noticed. A label alone leaves much unsaid.",
                "Duration adds useful context to the summary. It is not a feature of this model.",
                "A follow-up can suggest another symptom. Only an explicit confirmation adds it to the model.",
              ][context]
            }
          </p>
          <ChapterNote number="04" title="More context. Clear boundaries.">
            Questions should clarify, not manufacture evidence. The workspace
            asks a small set of deterministic follow-ups. It never turns a
            duration or a report value into a hidden symptom, and it does not
            send your answers to a language model.
          </ChapterNote>
        </div>
      </section>
      <section id="reports" data-chapter="4" className="chapter">
        <div className="chapter-inner">
          <ChapterMeta index={4} aside="Read. Extract. Review." />
          <h2>
            A page becomes
            <br />a starting point.
          </h2>
          <div className={`report-stage ${report ? "extracted" : ""}`}>
            <div className="paper">
              <span className="eyebrow">Fictional laboratory example</span>
              <h3>Sample 001</h3>
              <div>
                Hemoglobin <b>13.8 g/dL</b>
              </div>
              <div>
                Glucose <b>92 mg/dL</b>
              </div>
              <div>
                Platelets <b>250 ×10³/µL</b>
              </div>
              <p>
                Illustrative values only.
                <br />
                No real patient data.
              </p>
            </div>
            <span className="report-arrow">↗</span>
            <div className="structured-report">
              <span className="eyebrow">
                {report
                  ? "Editable structured findings"
                  : "Waiting for a closer look"}
              </span>
              {report ? (
                <>
                  <label>
                    Hemoglobin
                    <input
                      aria-label="Fictional hemoglobin value"
                      defaultValue="13.8"
                    />{" "}
                    <span>g/dL</span>
                  </label>
                  <label>
                    Glucose
                    <input
                      aria-label="Fictional glucose value"
                      defaultValue="92"
                    />{" "}
                    <span>mg/dL</span>
                  </label>
                  <p>
                    Source: page 1 · Fictional example
                    <br />
                    Check every value before using it.
                  </p>
                </>
              ) : (
                <div className="ghost-lines">
                  <i />
                  <i />
                  <i />
                </div>
              )}
            </div>
          </div>
          <div className="scene-caption">
            <button
              className="text-button"
              onClick={() => {
                setReport(!report);
                ping();
              }}
            >
              {report ? "Reset the example ↺" : "Reveal the structure ↗"}
            </button>
            <Link to="/workspace?tab=reports">Try the report reader ↗</Link>
          </div>
          <ChapterNote number="05" title="Extraction is not interpretation.">
            The live tool reads embedded PDF text first, then uses local OCR for
            scans and images. Some characters will be unclear. Missing values
            stay missing. You can review and correct the findings; lab values
            never change the symptom classifier.
          </ChapterNote>
        </div>
      </section>
      <section id="reasoning" data-chapter="5" className="chapter dark">
        <div className="chapter-inner">
          <ChapterMeta index={5} aside="Look inside a match" />
          <h2>
            Show your
            <br />
            reasoning.
          </h2>
          <div className="reasoning-scene">
            <div>
              <span className="eyebrow">Fictional case 01</span>
              <h3>{samples[0].assessment.matches[0].label}</h3>
              <p>What moved the model toward this label?</p>
              <Link to="/workspace?sample=cold">
                Open the complete example ↗
              </Link>
            </div>
            <div className="contribution-chart">
              {samples[0].assessment.matches[0].contributions
                .slice(0, 5)
                .map((c) => (
                  <div key={c.symptom_id}>
                    <span>
                      {c.label}
                      {!c.present ? " (not selected)" : ""}
                    </span>
                    <div className="bar-track">
                      <i
                        className={c.contribution < 0 ? "negative" : ""}
                        style={{
                          width: `${Math.min(100, Math.max(3, Math.abs(c.contribution) * 400))}%`,
                        }}
                      />
                    </div>
                    <small>
                      {c.contribution > 0 ? "+" : ""}
                      {c.contribution.toFixed(3)}
                    </small>
                  </div>
                ))}
            </div>
          </div>
          <ChapterNote
            number="06"
            title="An explanation of a model, not a person."
          >
            These are actual SHAP contributions for the predicted class from the
            bundled model. Positive contributions raise that class’s internal
            score; negative contributions lower it. This explains what the model
            did. It does not prove a cause, a diagnosis, or clinical
            reliability.
          </ChapterNote>
        </div>
      </section>
      <section id="limits" data-chapter="6" className="chapter limits dark">
        <div className="chapter-inner">
          <ChapterMeta index={6} aside="What remains outside the frame" />
          <div className="pin-scene">
            <h2>
              How much
              <br />
              can we know?
            </h2>
            <div
              className="tunnel"
              aria-hidden="true"
              style={{ "--depth": depth } as React.CSSProperties}
            >
              {Array.from({ length: 13 }, (_, i) => (
                <div
                  key={i}
                  style={{
                    transform: `translate(-50%,-50%) scale(${1 - i * 0.067}) rotateY(${(depth - 50) * 0.1}deg)`,
                    opacity: 1 - i * 0.056,
                  }}
                >
                  <span>uncertainty</span>
                </div>
              ))}
            </div>
          </div>
          <label className="limit-slider">
            Change your perspective
            <input
              aria-label="Tunnel perspective"
              type="range"
              value={depth}
              onChange={(e) => setDepth(+e.target.value)}
            />
          </label>
          <ChapterNote number="07" title="The limits belong in the foreground.">
            Only 304 unique symptom patterns. No documented clinical population.
            No prospective evaluation. Perfect performance on a small internal
            test does not make a medical tool. This project is an invitation to
            inspect the process, including everything it leaves out.
          </ChapterNote>
        </div>
      </section>
      <section id="explore" data-chapter="7" className="chapter explore">
        <div className="chapter-inner">
          <ChapterMeta index={7} aside="Your next question" />
          <h2>
            The next
            <br />
            <span className="blue">shape.</span>
          </h2>
          <p className="explore-lede">
            You have seen the ideas.
            <br />
            Now, put them to the test.
          </p>
          <div className="explore-links">
            <Link to="/workspace">
              <span>
                <small>01</small>Explore symptoms
              </span>
              <span>↗</span>
            </Link>
            <Link to="/workspace?tab=reports">
              <span>
                <small>02</small>Read a report
              </span>
              <span>↗</span>
            </Link>
            <Link to="/methods">
              <span>
                <small>03</small>Question the method
              </span>
              <span>↗</span>
            </Link>
          </div>
          <footer className="site-footer">
            <div>
              The Shape of Health
              <br />
              <span className="muted">
                An educational experiment by Archit.
              </span>
            </div>
            <Link to="/privacy">Privacy ↗</Link>
            <a href="https://github.com/Archit-k20/AI-Powered-Health-Prediction-Web-App">
              Source code ↗
            </a>
          </footer>
        </div>
      </section>
      <div className="bottom-bar">
        <button
          ref={menuButton}
          onClick={() => setMenu(true)}
          aria-haspopup="dialog"
          aria-expanded={menu}
        >
          <span className="bar-mark">╎╎╎╎╎</span>Chapters{" "}
          <small>{String(active + 1).padStart(2, "0")} / 08</small>
        </button>
        <div>
          <button
            role="switch"
            aria-checked={sound}
            onClick={() => setSound(!sound)}
          >
            ╎ Sound {sound ? "on" : "off"}
          </button>
          <button
            role="switch"
            aria-checked={motion}
            onClick={() => setMotion(!motion)}
          >
            Motion {motion ? "on" : "off"}
          </button>
        </div>
      </div>
      <dialog
        ref={dialog}
        className="chapter-dialog"
        onCancel={() => setMenu(false)}
        onClick={(e) => {
          if (e.target === e.currentTarget) setMenu(false);
        }}
      >
        <div className="dialog-top">
          <span>Explore the exhibition</span>
          <button onClick={() => setMenu(false)} aria-label="Close chapters">
            Close ×
          </button>
        </div>
        <div className="chapter-list">
          {chapters.map((c, i) => (
            <button key={c} onClick={() => jump(i)}>
              <small>0{i + 1}</small>
              <span>{c}</span>
              <span>↗</span>
            </button>
          ))}
        </div>
        <p>Eight chapters. One open experiment.</p>
        <button
          className="text-button"
          onClick={() => {
            setMenu(false);
            navigate("/workspace");
          }}
        >
          Go straight to the demo ↗
        </button>
      </dialog>
    </main>
  );
}
