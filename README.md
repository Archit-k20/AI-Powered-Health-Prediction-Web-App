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
- PDF/JPG/PNG extraction: embedded PDF text first, then local Tesseract OCR. Editable findings, manual additions, sequential queues, partial failures and cancellation.
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

Real symptoms and reports are processed by this application’s Python backend, deployed on Cloud Run in `us-central1`. They are **never sent to Gemini**. Deterministic follow-ups and summaries work without an API key. Health inputs stay in browser memory, and uploads are transient; application code does not log health content. Hosting providers may retain request metadata. Only the motion preference uses local storage.

The optional Gemini route accepts only a predefined fictional ID and rejects request bodies and query parameters. It receives fixed case content plus relevant source-linked catalog passages, and validates source IDs in its response. Missing credentials, quota limits or malformed responses fall back to a labeled prepared explanation. The default `GEMINI_MODEL` is `gemini-3.5-flash-lite`; `GEMINI_API_KEY` is server-side only.

## API

| Endpoint | Function |
|---|---|
| `GET /api/v1/health` | Readiness and capabilities |
| `GET /api/v1/symptoms` | Versioned vocabulary and follow-ups |
| `POST /api/v1/assessments` | Ranked matches and class-specific explanations |
| `POST /api/v1/lab-reports/extract` | One uploaded report |
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
