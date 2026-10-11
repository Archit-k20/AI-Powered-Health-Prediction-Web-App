import { Link } from "react-router-dom";
import info from "../../shared/model-info.json";
import catalog from "../../shared/catalog.json";
const percent = (x: number) => `${(x * 100).toFixed(1)}%`;
export default function Information({ page }: { page: string }) {
  return (
    <main id="main" className="information">
      <span className="eyebrow">ABOUT THE TOOLS / {page}</span>
      <h1>
        {page === "methods"
          ? "How the tools work"
          : page === "sources"
            ? "Sources you can read"
            : page === "about"
              ? "About The Shape of Health"
              : "Your privacy"}
      </h1>
      {page === "methods" && (
        <>
          <p className="information-lede">
            A transparent demonstration of a small model, and the boundaries of
            what it can tell us.
          </p>
          <div className="info-section">
            <h2>From duplicated rows to distinct patterns.</h2>
            <div>
              <p>
                The original preprocessing accidentally included disease names
                as features. This rebuild selects all 17 symptom columns and
                keeps the disease label exclusively as the target.
              </p>
              <p>
                The 4,920 rows reduce to 304 distinct symptom patterns, with no
                conflicting labels for identical patterns. A stratified split
                with seed 42 produces 243 training and 61 holdout examples.
                Identical patterns never appear on both sides.
              </p>
              <p>
                A random forest with 100 trees is compared with logistic
                regression. Four-fold cross-validation uses only the training
                partition. The deployed model is the evaluated model; the
                holdout is never used to fit it.
              </p>
            </div>
          </div>
          <div className="info-section">
            <h2>The measured results.</h2>
            <div className="metrics-table">
              <table>
                <caption>
                  Internal holdout of 61 patterns — not clinical performance
                </caption>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Macro F1</th>
                    <th>Top-three recall</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(info.evaluation).map(([name, m]) => (
                    <tr key={name}>
                      <td>{name.replace(/_/g, " ")}</td>
                      <td>{percent(m.accuracy)}</td>
                      <td>{m.macro_f1.toFixed(3)}</td>
                      <td>{percent(m.top_three_recall)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p>
                Perfect scores on this small, templated dataset do not establish
                usefulness with real patients. Many examples differ by only one
                symptom. Exact deduplication does not remove that similarity.
              </p>
              <details>
                <summary>Show all class results</summary>
                <table>
                  <thead>
                    <tr>
                      <th>Dataset label</th>
                      <th>F1</th>
                      <th>Examples</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(info.evaluation.random_forest.per_class)
                      .filter(
                        ([, v]) => typeof v === "object" && "precision" in v,
                      )
                      .map(([label, v]) => (
                        <tr key={label}>
                          <td>{label}</td>
                          <td>
                            {typeof v === "object" && "f1-score" in v
                              ? Number(v["f1-score"]).toFixed(3)
                              : ""}
                          </td>
                          <td>
                            {typeof v === "object" && "support" in v
                              ? Number(v.support)
                              : ""}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </details>
            </div>
          </div>
          <div className="info-section">
            <h2>How a match is explained.</h2>
            <div>
              <p>
                Selected symptoms become binary features. The model ranks 41
                dataset labels. SHAP contributions explain each specific class,
                including the effect of symptoms that were not selected.
              </p>
              <p>
                The largest contributions are shown in model-score units, never
                as your probability of disease. Feature order, class mapping,
                dataset hash and dependency versions are packaged with the model
                and checked at startup.
              </p>
              <p>
                Duration is context only. Suggested symptoms require
                confirmation. Lab values, documented diagnoses and BMI are
                separate from the model. Report-derived symptoms enter only
                after you confirm they are current.
              </p>
            </div>
          </div>
          <div className="info-section">
            <h2>What remains unknown.</h2>
            <ul>
              {info.limitations.map((l) => (
                <li key={l}>{l}</li>
              ))}
            </ul>
          </div>
          <div className="info-section">
            <h2>Inspect and reproduce.</h2>
            <div>
              <p>
                Model: {info.version}
                <br />
                Vocabulary: {info.registry_version}
              </p>
              <a href="https://github.com/Archit-k20/AI-Powered-Health-Prediction-Web-App">
                Training code and evaluation ↗
              </a>
              <p className="hash">Dataset SHA-256: {info.dataset_sha256}</p>
            </div>
          </div>
        </>
      )}
      {page === "sources" && (
        <>
          <p className="information-lede">
            Descriptions have sources. Predictions have limitations. One does
            not validate the other.
          </p>
          <p>
            The catalog contains brief original paraphrases of public
            educational material from MedlinePlus, the U.S. National Library of
            Medicine. These sources support general descriptions, not model
            rankings. Legacy dataset names have been corrected for readability.
          </p>
          <div className="source-list">
            {catalog.map((c) => (
              <div key={c.condition}>
                <h2>{c.label}</h2>
                <p>{c.description}</p>
                <a href={c.sources[0].url} target="_blank" rel="noreferrer">
                  MedlinePlus ↗
                </a>
              </div>
            ))}
          </div>
          <div className="info-section">
            <h2>Data and visual reference.</h2>
            <div>
              <p>
                The dataset is retained from{" "}
                <a href="https://github.com/Archit-k20/AI-Powered-Health-Prediction-Web-App">
                  Archit’s original college project
                </a>
                . Its upstream clinical provenance and collection method are
                undocumented.
              </p>
              <p>
                The interface takes inspiration from{" "}
                <a href="https://dribbble.com/tags/health-care-website">
                  healthcare website designs on Dribbble
                </a>
                . Health content, graphics and interactive code were made for
                this project. Typography is self-hosted Inter.
              </p>
            </div>
          </div>
        </>
      )}
      {page === "about" && (
        <>
          <p className="information-lede">
            The Shape of Health is an educational portfolio project by Archit,
            rebuilt from an AI-powered health prediction app made during
            college.
          </p>
          <div className="info-section">
            <h2>Why give it a new shape?</h2>
            <div>
              <p>
                The original question still matters: can we make model behavior
                understandable? This edition brings inputs, patterns,
                explanations and blind spots into one experience.
              </p>
              <p>
                The interface gives you a direct route to symptoms, report
                reading and downloadable summaries. Fictional examples help you
                try the tools without entering personal information.
              </p>
            </div>
          </div>
          <div className="info-section">
            <h2>What it is for.</h2>
            <div>
              <p>
                Learning, experimentation and conversations about responsible
                interface design. It is not a medical service. There are no
                accounts, prescriptions, consultations, bookings or stored
                health histories.
              </p>
              <p>
                Real health questions need qualified clinical care. This project
                has no clinical validation.
              </p>
              <Link to="/methods">Read the methodology ↗</Link>
            </div>
          </div>
          <div className="info-section">
            <h2>Made by Archit.</h2>
            <div>
              <a href="https://github.com/Archit-k20">GitHub ↗</a>
              <p>
                Visual inspiration:{" "}
                <a href="https://dribbble.com/tags/health-care-website">
                  healthcare website designs on Dribbble
                </a>
                .
              </p>
            </div>
          </div>
        </>
      )}
      {page === "privacy" && (
        <>
          <p className="information-lede">
            No account. No saved health history. A clear distinction between
            real inputs and fictional examples.
          </p>
          <div className="info-section">
            <h2>What happens in your browser.</h2>
            <div>
              <p>
                Symptom selections, answers, uploaded files, extracted text,
                confirmations and BMI values stay in application memory.
                Reloading or choosing “Clear session” clears them. They are not
                written to browser storage.
              </p>
              <p>
                Only the motion preference is stored locally. No analytics,
                advertising or tracking SDK is included. Three fictional cases
                and their prepared outputs work without the backend.
              </p>
            </div>
          </div>
          <div className="info-section">
            <h2>What goes to the server.</h2>
            <div>
              <p>
                Live symptom selections and answers are sent to this project’s
                Python backend on Google Cloud Run, in us-central1 (United
                States). Reports are sent there for local text extraction or OCR
                and rule-based clinical text extraction. No external AI service
                receives these documents.
              </p>
              <p>
                The server uses memory and short-lived OCR temporary files. It
                does not create a database or persistent upload history.
                Application code does not log request bodies, symptoms,
                extracted text or filenames. Hosting providers may retain
                operational metadata such as IP address, route, time and status
                under their policies.
              </p>
              <p>
                Vercel hosts the frontend. Report uploads go directly to the
                configured backend, without a frontend function proxy.
              </p>
            </div>
          </div>
          <div className="info-section">
            <h2>Gemini has a separate door.</h2>
            <div>
              <p>
                Your real symptoms, answers and reports are never sent to
                Gemini. Live follow-ups and summaries use deterministic
                templates. The optional Gemini endpoint accepts only three
                predefined fictional IDs; written text, query parameters and
                uploads are rejected.
              </p>
              <p>
                Only a fixed fictional case and catalog passages go to Gemini.
                Returned source IDs are checked. Missing credentials, quota
                errors or malformed responses fall back to a labeled prepared
                explanation.
              </p>
              <p>
                This follows{" "}
                <a href="https://ai.google.dev/gemini-api/terms">
                  Google’s restrictions on unpaid Gemini services
                </a>
                . API keys remain on the server.
              </p>
            </div>
          </div>
          <div className="info-section">
            <h2>Downloads are yours.</h2>
            <div>
              <p>
                The PDF is created in your browser when you choose to download
                it. You control where it is stored or shared.
              </p>
              <p>
                Use fictional walkthroughs for the lowest exposure. Upload
                reports only if you are comfortable with the server processing
                described here.
              </p>
            </div>
          </div>
        </>
      )}
      <footer className="information-footer">
        <Link to="/">← Back to home</Link>
        <Link to="/workspace">Open the tools →</Link>
      </footer>
    </main>
  );
}
