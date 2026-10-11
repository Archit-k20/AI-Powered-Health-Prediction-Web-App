# Testing and verification

Run commands from the repository root unless noted otherwise. Use Python 3.12 and Node.js 24, matching CI. Install `training/requirements.txt` into `.venv` and run `npm ci` in `frontend/` first.

## Automated checks

```sh
.venv/bin/python -m pytest backend/tests -q
npm --prefix frontend run types
npm --prefix frontend test
npm --prefix frontend run build
```

Real OCR cases require the `tesseract` executable with English language data. Those tests skip when it is absent; a passing run with skips does not verify OCR. The [CI workflow](../.github/workflows/verify.yml) installs Tesseract on Linux and runs both application suites.

| Area                    | Coverage                                                                                                                                  |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Training and artifacts  | Explicit symptom features, disjoint deduplicated splits, checksums, registry and class alignment                                          |
| Assessments             | Aliases, duplicates, unsupported or insufficient input, confirmed follow-ups, correct-class SHAP contributions                            |
| Reports                 | Embedded text, scanned PDFs, images, mixed PDF pages, malformed and oversized files, missing fields, table layouts, qualitative values    |
| Clinical mentions       | Negation, history, family and hypothetical context, ambiguous fever, conflicting statements, unsupported phrases                          |
| Input confirmation      | Explicit current-symptom confirmation, revocation, cross-document conflicts, rejection of unconfirmed API inputs                          |
| Reliability and privacy | Queue bounds, deadlines, cancellation, model failures, redacted errors, no health-data logging, isolation from Gemini                     |
| Frontend                | Offline examples, backend failure states, report queues and partial failure, editing, keyboard tabs, PDF pagination and source provenance |

All files under `backend/tests/fixtures/` are fictional. Some tests create additional synthetic documents in memory. Keep user reports, exported personal summaries, credentials, and request payload dumps out of fixtures and screenshots.

## Reproducibility and generated files

```sh
.venv/bin/python training/train.py
.venv/bin/python training/export_samples.py
npm --prefix frontend run types
git diff --exit-code -- backend/models shared ":(exclude)shared/samples.json"
.venv/bin/python training/verify_samples.py
git diff --exit-code -- frontend/src/generated/api.ts docs/EVALUATION.md
```

Run these against a committed reference when checking reproducibility. `training/verify_samples.py` compares generated examples with `HEAD:shared/samples.json`, allowing an absolute tolerance of `1e-10` for numerical SHAP variation across CPU architectures. Structure, ranks, and sources must remain identical. Model artifacts, registry, metrics, and other generated contracts are checked without that tolerance.

`training/catalog.py` regenerates the condition catalog from its maintained source text. `training/create_fixtures.py` regenerates the fictional report files. Neither is needed for routine startup. Review changes to generated files before committing them.

## Browser smoke test

1. Open the homepage and enter each workspace tab using both navigation and direct URLs. Check keyboard focus and reduced-motion behaviour.
2. With no backend URL configured, open each fictional symptom case and both fictional report examples. Verify the interface labels prepared data and unavailable live tools honestly.
3. Connect the backend. Choose at least two symptoms, edit follow-up answers, confirm any suggested additions, and request matches. Inspect a supporting explanation and its source link.
4. Upload the fictional text PDF, scanned PDF, and image from `backend/tests/fixtures/`. Check sequential progress, the original preview, source links, and editable values. Missing results such as `TSH pending` must remain blank.
5. Open the fictional doctor note. Cough and fatigue require confirmation; denied shortness of breath, hypothetical nausea, and historical migraine remain document context. Confirm, then revoke a symptom and check the selected inputs.
6. Export a summary and inspect every PDF page for clipping, page breaks, inputs, reviewed findings, evidence, and the educational-use notice. Built-in PDF fonts transliterate scientific units and unsupported characters; compare the source when exact typography matters.
7. Test a malformed file, a failed request, and a cancelled extraction. Failures must leave an actionable state and must not produce a fabricated prediction.
8. Check narrow-screen layouts for horizontal overflow, readable controls, and accessible report previews. Use only fictional examples in any published captures.

## What these checks do not establish

Regression coverage verifies specified software behaviour. It does not measure clinical accuracy or comprehensive extraction quality across real reports. The model uses a small, repetitive dataset with undocumented clinical provenance. Report rules can miss unfamiliar language, contextual boundaries, complex layouts, and poor scans. There is no representative labelled patient-report benchmark. See [model evaluation](EVALUATION.md) and [deployment operations](DEPLOYMENT.md) for the remaining model and service limits.
