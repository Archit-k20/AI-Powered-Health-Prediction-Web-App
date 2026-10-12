# Deployment and operations

Frontend: Vercel Hobby (noncommercial portfolio). API: Google Cloud Run, project `shape-health-demo-2026`, region `us-central1`.

## Prerequisites

Sign in with `gcloud auth login` and `npx vercel login`. When multiple Google accounts are signed in, pass `--account=YOUR_GOOGLE_EMAIL` on GCP commands and confirm that account owns the intended project. Use a dedicated GCP project linked to an open billing account. Do not deploy into an unrelated application project. The checked-in code includes no credentials.

Enable `run.googleapis.com`, `artifactregistry.googleapis.com`, `iam.googleapis.com`, and `billingbudgets.googleapis.com`. Create a Docker Artifact Registry repository named `health-demo` and a dedicated runtime service account with no additional IAM roles. The runtime needs no access to GCP data services.

## Backend

Build an AMD64 image from the repository root. Cloud Run cannot use the ARM-only image produced by a default Apple Silicon build.

```sh
gcloud auth configure-docker us-central1-docker.pkg.dev
docker buildx build --platform linux/amd64 -f backend/Dockerfile.backend \
  -t us-central1-docker.pkg.dev/shape-health-demo-2026/health-demo/api:v3 --push .
gcloud run deploy health-api --project shape-health-demo-2026 \
  --region us-central1 \
  --image us-central1-docker.pkg.dev/shape-health-demo-2026/health-demo/api:v3 \
  --service-account health-runtime@shape-health-demo-2026.iam.gserviceaccount.com \
  --cpu 1 --memory 1Gi --min 0 --max 1 --min-instances 0 --max-instances 1 \
  --concurrency 4 --timeout 120 --cpu-throttling --no-cpu-boost \
  --startup-probe httpGet.path=/api/v1/health,httpGet.port=8080,periodSeconds=10,timeoutSeconds=5,failureThreshold=12 \
  --allow-unauthenticated \
  --set-env-vars ALLOWED_ORIGINS=https://the-shape-of-health.vercel.app,GEMINI_MODEL=gemini-3.5-flash-lite
```

Set exact origins; wildcard CORS is rejected by configuration. Native Cloud Run request metadata may be retained by Google; do not enable request body logging or an APM integration that captures bodies. Application access logs are disabled. There are no startup network downloads.

The app allows one active extraction and up to two waiting extractions, with a 105-second total queue/processing bound and a 90-second processing deadline. Each OCR text pass is limited to 18 seconds. Orientation detection is capped at five seconds; uncertain pages may receive one alternate-layout OCR pass, within the same 90-second document deadline. Disconnection kills the worker process group. Requests are limited per peer and globally in memory. These limits complement, but do not replace, Cloud Run’s max instance and request limits.

Core features require **no Gemini key**. For optional fictional explanations, store a key in Secret Manager and grant the runtime account access to that single secret, then configure `GEMINI_API_KEY` as a secret environment reference. Do not put the key in frontend variables, repository files, build arguments or deployment command text. Set `GEMINI_MODEL` separately if needed.

## Frontend

From `frontend/`, link the Vercel project and set `VITE_API_URL` to the verified Cloud Run URL for production and preview environments. The deployed URL is `https://health-api-1080173394372.us-central1.run.app`. Vite embeds this public endpoint at build time; changing it requires a new build.

```sh
npm ci
npm test
npx vercel link --project the-shape-of-health
npx vercel env add VITE_API_URL production
npx vercel pull --yes --environment=production
npx vercel build --prod
npx vercel deploy --prebuilt --prod
```

Build locally and deploy the prebuilt output from `frontend/` so the shared JSON files one directory above are available to the local build. If connecting the monorepo through Vercel’s Git integration, set Root Directory to `frontend` and enable inclusion of source files outside the root directory. `frontend/vercel.json` provides SPA route rewrites. No Vercel serverless function is used for uploads.

## Costs and cleanup

Use request-based billing, min 0, max 1, no CPU boost, no database and no startup embedding downloads. The deployed project has a ₹100 monthly budget. Configure a project-filtered monthly budget with alerts at 50%, 90% and 100%. Budget alerts are not spending caps. Avoid frequent model/image pushes; retain one current image and a short-lived rollback image, then delete only obsolete images in this project’s repository. Artifact storage, egress, builds and other provider resources may incur charges even when request usage is within a free allowance.

Official references: [Cloud Run pricing](https://cloud.google.com/run/pricing), [budget behavior](https://docs.cloud.google.com/billing/docs/how-to/budgets), [Vercel Hobby](https://vercel.com/docs/plans/hobby), [Gemini terms](https://ai.google.dev/gemini-api/terms).

## Smoke check

1. `GET /api/v1/health` reports `ready`, assessments true and OCR true.
2. All three examples work with the backend disconnected.
3. A live assessment returns matches with a model version and sources.
4. Upload the fictional text PDF, scanned PDF and PNG in `backend/tests/fixtures` sequentially. Correct a value and download a PDF.
5. Check privacy, methodology and direct `/workspace?tab=reports` links.
6. Verify CORS from the production domain and rejection of a different origin.
7. Confirm the deployed instance limits and no public secrets. Leave Gemini unconfigured if optional AI output is not required.

## If live tools become unavailable

Check the API health endpoint and Cloud Run error logs before assuming a cold start. A service can still show a ready revision while Google rejects requests because its billing account is closed. Check both the project's billing link and the linked account's `open` status; a project-level `billingEnabled: true` alone is insufficient.

The current project is `shape-health-demo-2026`. After a project migration, update `VITE_API_URL` in Vercel and the `connect-src` origin in `frontend/vercel.json`, then rebuild and deploy. Verify a live assessment and fictional report extraction from the production origin. The previous project is not a fallback if its billing account is closed.
