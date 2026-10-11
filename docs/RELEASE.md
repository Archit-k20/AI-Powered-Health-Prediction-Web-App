# Release record — 12 October 2026

## Public deployment

- **Demo:** https://the-shape-of-health.vercel.app
- **Workspace:** https://the-shape-of-health.vercel.app/workspace
- **API readiness:** https://health-api-888697615614.us-central1.run.app/api/v1/health
- Frontend: Vercel project `the-shape-of-health`, verified **Hobby** plan. Production alias is publicly accessible without authentication.
- Backend: separate GCP project `shape-health-archit-2026`, Cloud Run `health-api`, `us-central1`, revision `health-api-00003-q7n`.
- Image: `us-central1-docker.pkg.dev/shape-health-archit-2026/health-demo/api:v3`; manifest digest `sha256:8a57da6331614c999cbe2df56a4aeceebaca44201f68cc4f8df8defd997e7189`.
- One vCPU, 1 GiB RAM, one application worker, minimum zero / maximum one instance at both service and revision level, concurrency four, request timeout 120 seconds, request-based CPU allocation. HTTP startup readiness probe is enabled.
- Dedicated runtime service account with no additional data-service roles. No database or persistent upload bucket. Model, catalog and OCR runtime are bundled.
- ₹100/month project budget with 50%, 90% and 100% alerts. The goal is ₹0 under free allowances. **Alerts do not cap spending.**

## Verified working

- Revised white/soft-teal healthcare homepage with direct symptom/report entry points, familiar navigation, readable forms and restrained transitions. Common symptoms appear first; fictional cases and technical explanations expand when needed. Desktop and 390 px mobile checks are included.
- Three fictional symptom walkthroughs work without a backend. Live symptom assessments return ranked matches, actual class-specific SHAP contributions, model version and source links.
- Public browser-to-Cloud-Run assessment and sequential text PDF / scanned PDF / PNG extraction. Each synthetic report produced four findings; deliberately missing fields remained blank. Edited values appeared in the downloaded four-page PDF.
- Backend public API checks cover all three sample patterns, embedded-text extraction, OCR, prepared explanation fallback and rejection of visitor-written content at the fictional explanation endpoint. See `public-api-checks.json`.
- Backend readiness reports assessment, PDF text and OCR capabilities. Production CORS origin is accepted; an unrelated origin receives no permission.
- All six frontend routes return HTTP 200 without authentication. Public browser checks show no application console errors. Mobile layouts have no horizontal page overflow.
- 71 backend tests pass with actual Tesseract inside the Linux runtime. Twenty-one frontend interaction/export/fallback tests pass. TypeScript/Vite production build succeeds.
- Repeated training produces identical split hashes, metrics and artifact checksum locally. Both random forest and logistic baseline are evaluated; all measured results and limits are in `EVALUATION.md`.
- Runtime dependency and npm audits found no known vulnerabilities at verification time. All 41 catalog source URLs were reachable; see `source-check.json`.

## Provider and product limits

Gemini has **no key configured**. Its optional fictional-example button returns a clearly labeled prepared explanation. No live Gemini generation is claimed. Core tools are fully functional without it, and real visitor inputs cannot be submitted to that endpoint.

During the earlier release, a newly created Cloud Run instance took approximately 30 seconds to reach the readiness probe during initial deployment. Scale-to-zero is configured. A later public browser check also verified idle wake-up: the UI displayed its connecting state and the queued assessment completed successfully. System logs measured about 10 seconds from that autoscaling instance start to readiness. Idle wake-up latency varies; the UI preserves fictional examples if a request times out. The API does not receive a keep-alive ping schedule.

The model is not clinically validated. Perfect scores on 61 holdout patterns from a repetitive historical dataset do not establish clinical performance. Dataset provenance and population remain undocumented.

The report reader recognizes 61 named lab analytes plus unambiguous rows under recognized table headers. It also extracts supported clinical symptom mentions and separate document claims. English laboratory reports, notes, discharge summaries and written radiology reports are handled on a best-effort basis. The rules can miss unfamiliar wording, complex tables, context boundaries or poor scans. Handwriting and non-English text are not reliably supported. No representative clinical extraction benchmark has been completed. Uncertain fields remain reviewable, and low-quality scans cannot transfer symptom inputs.

Production was deployed from local prebuilt output. Automatic Git-to-Vercel deployment is not connected; repeat the documented commands for future releases. Credentials remain in the user's hosting tools and ignored local configuration, never in the repository. See `DEPLOYMENT.md` for maintenance, redeployment and cost controls.

## Included artifacts

- `docs/EVALUATION.md` — measured evaluation and per-class results
- `docs/VERIFICATION.md` — test coverage and known product limitations
- `docs/DEPLOYMENT.md` — deployment and operations instructions
- `docs/cloud-run-config.json` — sanitized resource configuration
- `docs/screenshots/clinical-*` — public demo desktop/mobile captures
- `docs/example-summary.pdf` — actual browser-exported fictional summary

## Healthcare interface revision

The exhibition was replaced following user feedback. At that revision the core model, API, privacy boundary, report parser and hosting configuration were unchanged. The later report upgrade is described below. Three.js, GSAP, procedural sound and the eight-chapter navigation were removed. The main JavaScript entry decreased from about 140 kB to 86 kB compressed; the separate 132 kB compressed sculpture bundle is no longer needed. These are build sizes, not a measured page-speed score.


## Report review revision

- Frontend deployment: `dpl_Kd58kDqueYKHAEh63pn521HU4Ght`, publicly aliased to the existing demo URL.
- Pipeline `reports-2.0.0`: page classification, positioned text/table parsing, scanned-body fallback, orientation checks, OCR quality warnings and exact source quotes.
- Report symptoms require explicit current-symptom confirmation. The API rejects unconfirmed report inputs. Lab results and documented diagnoses/causes never become invented model features. Conflicting mentions require manual clarification.
- Original PDFs render in a self-hosted lazy-loaded PDF.js canvas, including browsers without native PDF support. Evidence coordinates support source highlights. Preview rendering stays in the browser.
- Public API verification passed for report extraction, confirmed symptom assessment, unconfirmed-input rejection and production CORS. A user-provided scanned blank template was uploaded through the public browser and returned zero findings/mentions with an unfilled-template message and OCR warnings. Its content is not committed.
- Local browser checks passed for the complete confirmation-to-assessment workflow, 390-pixel mobile layout and a four-page PDF export retaining evidence and confirmation provenance. The original model artifacts and clinical-validity limitations are unchanged.
- The first measured public OCR request for the low-quality template completed in about 31 seconds. Dense scans may be slower and remain subject to bounded deadlines; no speed guarantee is made.
