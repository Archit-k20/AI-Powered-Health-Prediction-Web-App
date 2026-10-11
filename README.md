# The Shape of Health

An educational health project by **Archit**, with a calm, accessible interface for symptom exploration, report extraction and downloadable summaries. The homepage leads directly to the tools.

**[Open the live demo →](https://the-shape-of-health.vercel.app)** · [Open the workspace](https://the-shape-of-health.vercel.app/workspace) · [Release record](docs/RELEASE.md)

**Educational only. Not clinically validated. No diagnosis, prescribing, accounts, bookings or saved health history.**

## Explore

- React + TypeScript + Vite, custom CSS, self-hosted Inter.
- White and soft-teal healthcare layout, clear forms, responsive cards and minimal transitions. No WebGL, sound or scroll-driven exhibition.
- 131 supported symptom features with canonical IDs and search aliases.
- Three fictional walkthroughs with real precomputed model outputs; no backend required.
- Live random-forest matches, class-specific SHAP contributions and attributed descriptions.
- PDF/JPG/PNG report review: positioned PDF text and local Tesseract OCR, table-column parsing, source quotes, original-document preview, page quality and document-type detection. English lab reports, clinical notes, discharge summaries and written radiology reports are supported on a best-effort basis.
- Clinical mentions retain negation, history, family and hypothetical context. Only explicitly confirmed current symptom mappings reach the classifier. Lab values, diagnoses and reported causes remain document context. Conflicting mentions require manual clarification.
- Editable lab findings, manual additions, sequential queues, partial failures, cancellation and bounded processing. Unfilled templates and unreadable pages return actionable empty states. Two fictional report examples work offline.
- Paginated PDF summaries created in the browser. Separate optional adult BMI arithmetic.
- Methodology, source catalog, about and privacy pages.

The interface takes inspiration from [healthcare website designs on Dribbble](https://dribbble.com/tags/health-care-website). It prioritizes direct access to symptoms and reports, common symptoms before the complete vocabulary, and expandable technical explanations. It does not offer diagnosis or treatment.

## Run locally

Python 3.12 and Node 24 are used. The evaluated model is already bundled; startup does not download models or train anything.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r training/requirements.txt
.venv/bin/python -m uvicorn backend.app:app --port 8080 --no-access-log
```

For OCR, install Tesseract with English language data, or use the container:

```sh
docker compose up --build
```

In another terminal:

```sh
cd frontend
npm ci
VITE_API_URL=http://localhost:8080 npm run dev
```

Default CORS permits `localhost:5173` and `127.0.0.1:5173`. Set `ALLOWED_ORIGINS` to the exact dev origin if Vite selects a different port. With no `VITE_API_URL`, the homepage and fictional examples work and live tools show an explicit unavailable state.

## What changed in the model

The original preprocessing included disease-name features and omitted the final symptom column. The rebuild selects only `Symptom_1` through `Symptom_17`, normalizes naming, deduplicates patterns **before** splitting, and trains a 100-tree random forest with seed 42.

4,920 raw rows reduce to **304 unique patterns**, with no conflicting labels. The stratified 80/20 split contains 243 training and 61 holdout patterns. Four-fold cross-validation runs only within training. Logistic regression provides a baseline. Both models achieve 1.0 accuracy, macro F1 and top-three recall on this tiny internal holdout; this reflects repetitive templates and is **not evidence of clinical accuracy**.

See [the measured evaluation](docs/EVALUATION.md), including every class and limitations. Model metadata records feature order, class mapping, split hashes, dataset hash, dependency versions and artifact checksum. Startup validates the packaged model and registry.

```sh
.venv/bin/python training/train.py
.venv/bin/python training/catalog.py
.venv/bin/python training/export_samples.py
npm --prefix frontend run types
```

The original notebooks remain historical material. **Do not use them to build the deployed model.** `training/train.py` is the current training entry point. The dataset’s upstream provenance and population remain undocumented.

## Privacy boundary

Real symptoms and reports are processed by this application’s Python backend, deployed on Cloud Run in `us-central1`. They are **never sent to Gemini**. Deterministic follow-ups and summaries work without an API key. Health inputs, original files and extracted transcripts stay in browser memory, and server uploads are transient; application code does not log health content. Hosting providers may retain request metadata. Only the motion preference uses local storage.

The optional Gemini route accepts only a predefined fictional ID and rejects request bodies and query parameters. It receives fixed case content plus relevant source-linked catalog passages, and validates source IDs in its response. Missing credentials, quota limits or malformed responses fall back to a labeled prepared explanation. The default `GEMINI_MODEL` is `gemini-3.5-flash-lite`; `GEMINI_API_KEY` is server-side only.

## API

| Endpoint | Function |
|---|---|
| `GET /api/v1/health` | Readiness and capabilities |
| `GET /api/v1/symptoms` | Versioned vocabulary and follow-ups |
| `POST /api/v1/assessments` | Ranked matches and class-specific explanations |
| `POST /api/v1/reports/extract` | One document → lab rows, source-backed clinical mentions, page quality and warnings |
| `GET /api/v1/model-info` | Evaluation and limitations |
| `POST /api/v1/samples/{sample_id}/explain` | Allowlisted fictional example only |

Pydantic responses generate `shared/openapi.json` and `frontend/src/generated/api.ts`. No legacy API compatibility is retained.

## Verification

```sh
.venv/bin/python -m pytest backend/tests -q
npm --prefix frontend test
npm --prefix frontend run build
```

Real OCR tests require Tesseract. Fixtures contain fictional data only. Tests cover leakage, artifact integrity, aliases, follow-up corrections, SHAP class indexing, report formats and limits, bounded queues, cancellation, privacy isolation, provider fallback, frontend interactions and PDF pagination. See [verification notes](docs/VERIFICATION.md), [screenshots](docs/screenshots), and [a browser-exported fictional PDF](docs/example-summary.pdf).

## Hosting

Vercel Hobby serves the frontend; Cloud Run hosts the Python API in a separate GCP project. Browser API calls go directly to Cloud Run. One vCPU, 1 GiB, one worker, min 0/max 1 instances, request concurrency 4, timeout 120 seconds; only one extraction runs at a time.

See [deployment instructions](docs/DEPLOYMENT.md). Target cost is ₹0 through scale-to-zero and conservative limits. Free allowances and billing alerts are **not a hard spending cap**; small charges remain possible. No database, GPU, external load balancer or persistent upload bucket is used.


### Report review pipeline

`reports-2.0.0` reads positioned PDF text, checks image regions even when a PDF has selectable headers, and uses bounded local OCR with orientation checks and a fallback segmentation pass. It classifies pages and extracts source-linked lab rows and clinical mentions. The original document and a reading transcript remain available beside the review form. A self-hosted, lazy-loaded PDF.js renderer displays pages without a browser PDF plugin; selecting evidence highlights its source region where coordinates are available.

Visitors must confirm each eligible report symptom as current. `POST /api/v1/assessments` accepts `report_symptoms: [{symptom_id, confirmed_current: true}]` in addition to manual symptoms; the backend rejects unconfirmed and unsupported inputs. Removing a report or deselecting a symptom revokes its contribution. Contradictory mentions are not resolved by guessing dates or taking a majority vote. The previous lab-only URL remains as a hidden transition alias for already-open clients.

Lab values, documented diagnoses, historical/family mentions and attributed causes never become inferred symptoms. A generic fever mention is left unmapped until the visitor chooses the appropriate severity. The classifier and its 304-pattern evaluation are unchanged; this upgrade improves input review, not clinical validity. Extraction uses bounded, deterministic rules, not comprehensive medical understanding. Read all uncertain fields against the original; unsupported layouts and blank forms can legitimately produce no model inputs.
