import { Link } from "react-router-dom";
import { Icon } from "./Icon";

export const examples = [
  {
    id: "cold",
    title: "Cold-like symptoms",
    description: "Sneezing, a runny nose and a cough.",
    icon: "symptoms" as const,
  },
  {
    id: "migraine",
    title: "Headache symptoms",
    description: "A headache with changes in vision.",
    icon: "heart" as const,
  },
  {
    id: "skin",
    title: "Skin symptoms",
    description: "An itchy rash and changes to the skin.",
    icon: "shield" as const,
  },
];

export default function Home() {
  return (
    <main id="main" className="home">
      <section className="care-hero">
        <div className="care-hero-inner">
          <div className="care-intro">
            <span className="care-label">
              <span /> Health information, made approachable
            </span>
            <h1>
              A clearer picture.
              <br />
              <em>A simpler start.</em>
            </h1>
            <p>
              Explore symptoms, organise your report findings, and bring your
              questions together in one easy-to-use space.
            </p>
            <div className="hero-actions">
              <Link className="button-primary" to="/workspace">
                Explore symptoms <Icon name="arrow" />
              </Link>
              <Link className="button-secondary" to="/workspace?tab=reports">
                Read a report
              </Link>
            </div>
            <div className="hero-reassurance">
              <Icon name="check" /> No account needed <span>·</span> Free to
              explore
            </div>
            <p className="hero-notice">
              An educational project, not a diagnosis or treatment service.
            </p>
          </div>
          <div className="start-card">
            <div className="start-card-heading">
              <span className="icon-tile">
                <Icon name="heart" />
              </span>
              <div>
                <span className="eyebrow">LET’S START WITH YOU</span>
                <h2>What brings you here?</h2>
              </div>
            </div>
            <Link className="start-option" to="/workspace">
              <span className="icon-tile">
                <Icon name="symptoms" />
              </span>
              <span>
                <strong>I want to explore symptoms</strong>
                <small>
                  Choose what you’ve noticed and review possible matches.
                </small>
              </span>
              <Icon name="arrow" />
            </Link>
            <Link className="start-option" to="/workspace?tab=reports">
              <span className="icon-tile blue-tile">
                <Icon name="report" />
              </span>
              <span>
                <strong>I have a medical report</strong>
                <small>
                  Turn a PDF or image into findings you can review and edit.
                </small>
              </span>
              <Icon name="arrow" />
            </Link>
            <div className="start-card-foot">
              <Icon name="shield" />
              <span>
                Your session isn’t saved.{" "}
                <Link to="/privacy">How your data is processed</Link>
              </span>
            </div>
          </div>
        </div>
      </section>
      <section className="care-principles" aria-label="What to expect">
        <div>
          <Icon name="check" />
          <span>
            <strong>Clear, guided steps</strong>
            <small>Start with what you know.</small>
          </span>
        </div>
        <div>
          <Icon name="report" />
          <span>
            <strong>Information you can review</strong>
            <small>Edit findings and see their sources.</small>
          </span>
        </div>
        <div>
          <Icon name="shield" />
          <span>
            <strong>No saved health history</strong>
            <small>Your inputs stay in this session.</small>
          </span>
        </div>
      </section>
      <section className="care-section how-it-works" id="how-it-works">
        <div className="care-section-heading">
          <div>
            <span className="eyebrow">A LITTLE GUIDANCE</span>
            <h2>Three steps to a clearer summary.</h2>
          </div>
          <Link to="/methods">
            How the tools work <Icon name="arrow" />
          </Link>
        </div>
        <div className="care-steps">
          <article>
            <span className="step-number">1</span>
            <h3>Start with your information</h3>
            <p>
              Select symptoms or upload a report. Use a fictional example if you
              just want to look around.
            </p>
          </article>
          <article>
            <span className="step-number">2</span>
            <h3>Review what comes back</h3>
            <p>
              Read the model’s possible matches, check the sources, or correct
              the extracted report values.
            </p>
          </article>
          <article>
            <span className="step-number">3</span>
            <h3>Keep it all together</h3>
            <p>
              Download a summary of your inputs and findings to support a
              conversation with a healthcare professional.
            </p>
          </article>
        </div>
      </section>
      <section className="example-section" id="examples">
        <div className="care-section">
          <div className="care-section-heading">
            <div>
              <span className="eyebrow">TAKE A LOOK FIRST</span>
              <h2>Try it with an example.</h2>
              <p>Three fictional cases. No personal information needed.</p>
            </div>
            <span className="subtle-pill">
              Works even when live tools are offline
            </span>
          </div>
          <div className="example-cards">
            {examples.map((example) => (
              <Link
                key={example.id}
                to={`/workspace?sample=${example.id}`}
                className="example-card"
              >
                <span className="icon-tile">
                  <Icon name={example.icon} />
                </span>
                <h3>{example.title}</h3>
                <p>{example.description}</p>
                <span className="example-link">
                  Open fictional example <Icon name="arrow" />
                </span>
              </Link>
            ))}
          </div>
        </div>
      </section>
      <section className="care-section care-faq">
        <div>
          <span className="eyebrow">GOOD TO KNOW</span>
          <h2>A few things before you start.</h2>
          <p>
            The tools are here to help you explore information. Decisions about
            treatment belong with a qualified healthcare professional.
          </p>
          <Link className="text-link" to="/about">
            About this project <Icon name="arrow" />
          </Link>
        </div>
        <div className="faq-list">
          <details>
            <summary>Can this tell me which treatment I need?</summary>
            <p>
              No. It shows educational matches from a small symptom dataset. It
              cannot diagnose a condition, prescribe treatment, or assess
              urgency. If symptoms are severe, sudden or worsening, seek medical
              care.
            </p>
          </details>
          <details>
            <summary>What happens to my symptoms and reports?</summary>
            <p>
              Live inputs are processed by this project’s Python backend on
              Google Cloud Run. They are not sent to Gemini or saved as a health
              history. Your browser session clears when you reload or choose
              Clear session.{" "}
              <Link to="/privacy">Read the privacy details.</Link>
            </p>
          </details>
          <details>
            <summary>Can I use this without uploading anything?</summary>
            <p>
              Yes. Open any fictional example to try symptom results and
              explanations. The report tool also includes a fictional report you
              can edit and export.
            </p>
          </details>
        </div>
      </section>
      <footer className="care-footer">
        <div>
          <strong>The Shape of Health</strong>
          <p>An educational project by Archit.</p>
        </div>
        <nav aria-label="Footer">
          <Link to="/methods">Methodology</Link>
          <Link to="/sources">Sources</Link>
          <Link to="/privacy">Privacy</Link>
          <Link to="/about">About</Link>
        </nav>
        <small>For learning and exploration. Not clinically validated.</small>
      </footer>
    </main>
  );
}
