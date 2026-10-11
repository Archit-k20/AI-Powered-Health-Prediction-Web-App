# Release record — 11 October 2026

## Public deployment

- **Demo:** https://the-shape-of-health.vercel.app
- **Workspace:** https://the-shape-of-health.vercel.app/workspace
- **API readiness:** https://health-api-888697615614.us-central1.run.app/api/v1/health
- Frontend: Vercel project `the-shape-of-health`, verified **Hobby** plan. Production alias is publicly accessible without authentication.
- Backend: separate GCP project `shape-health-archit-2026`, Cloud Run `health-api`, `us-central1`, revision `health-api-00002-fgz`.
- Image: `us-central1-docker.pkg.dev/shape-health-archit-2026/health-demo/api:v2`; manifest digest `sha256:78038fe021c71dd70b4841dc38c2ddaec6a86c80059f4f57539cefc50bd28dd4`.
- One vCPU, 1 GiB RAM, one application worker, minimum zero / maximum one instance at both service and revision level, concurrency four, request timeout 120 seconds, request-based CPU allocation. HTTP startup readiness probe is enabled.
- Dedicated runtime service account with no additional data-service roles. No database or persistent upload bucket. Model, catalog and OCR runtime are bundled.
- ₹100/month project budget with 50%, 90% and 100% alerts. The goal is ₹0 under free allowances. **Alerts do not cap spending.**

## Verified working

- Eight-chapter responsive story, custom ribbon sculpture, shape controls, chapter navigation, motion control and opt-in procedural sound. Desktop and 390 px mobile reference comparisons and screenshots are included.
- Three fictional symptom walkthroughs work without a backend. Live symptom assessments return ranked matches, actual class-specific SHAP contributions, model version and source links.
- Public browser-to-Cloud-Run assessment and sequential text PDF / scanned PDF / PNG extraction. Each synthetic report produced four findings; deliberately missing fields remained blank. Edited values appeared in the downloaded four-page PDF.
- Backend public API checks cover all three sample patterns, embedded-text extraction, OCR, prepared explanation fallback and rejection of visitor-written content at the fictional explanation endpoint. See `public-api-checks.json`.
- Backend readiness reports assessment, PDF text and OCR capabilities. Production CORS origin is accepted; an unrelated origin receives no permission.
- All six frontend routes return HTTP 200 without authentication. Public browser checks show no application console errors. Mobile layouts have no horizontal page overflow.
- 32 backend tests pass with actual Tesseract inside the Linux runtime. Thirteen frontend interaction/export/fallback tests pass. TypeScript/Vite production build succeeds.
- Repeated training produces identical split hashes, metrics and artifact checksum locally. Both random forest and logistic baseline are evaluated; all measured results and limits are in `EVALUATION.md`.
- Runtime dependency and npm audits found no known vulnerabilities at verification time. All 41 catalog source URLs were reachable; see `source-check.json`.

## Provider and product limits

Gemini has **no key configured**. Its optional fictional-example button returns a clearly labeled prepared explanation. No live Gemini generation is claimed. Core tools are fully functional without it, and real visitor inputs cannot be submitted to that endpoint.

A newly created Cloud Run instance took approximately 30 seconds to reach the readiness probe during initial deployment. Scale-to-zero is configured. Idle wake-up latency varies and has not been separately benchmarked; the UI shows a starting/unavailable message and preserves fictional examples if a request times out. The API does not receive a keep-alive ping schedule.

The model is not clinically validated. Perfect scores on 61 holdout patterns from a repetitive historical dataset do not establish clinical performance. Dataset provenance and population remain undocumented.

The report parser supports 22 common analytes and may partially extract unfamiliar layouts. OCR can miss scientific units; the scanned fixture intentionally demonstrates a blank unit requiring review. This is extraction, not interpretation. Unsupported PDF-font characters are transliterated in exports.

Production was deployed from local prebuilt output. Automatic Git-to-Vercel deployment is not connected; repeat the documented commands for future releases. Credentials remain in the user's hosting tools and ignored local configuration, never in the repository. See `DEPLOYMENT.md` for maintenance, redeployment and cost controls.

## Included artifacts

- `docs/EVALUATION.md` — measured evaluation and per-class results
- `docs/VERIFICATION.md` — test coverage and known product limitations
- `docs/DEPLOYMENT.md` — deployment and operations instructions
- `docs/cloud-run-config.json` — sanitized resource configuration
- `docs/screenshots/public-*` — public demo desktop/mobile captures
- `docs/example-summary.pdf` — actual browser-exported fictional summary
