# Verification record

Local verification on 2026-10-11:

- 32 backend tests pass inside the Linux runtime with Tesseract. Native macOS runs skip the three OCR cases when Tesseract is absent.
- Thirteen frontend tests pass, covering all offline examples, aliases, duplicate removal, follow-up confirmation and corrections, backend failure, editable reports, sequential three-file processing, partial failure, keyboard tabs, memory-only data, PDF pagination, direct homepage entry points, common-symptom browsing and expandable model explanations.
- Production TypeScript and Vite build passes. Workspace and PDF code load separately; Three.js and GSAP have been removed.
- Retraining reproduces model checksum, split hashes and measured metrics. Linux CI reproduced the exact model checksum and metrics; exported SHAP decimals differed by approximately 1e-13 across CPU architectures, so example comparisons allow absolute numerical error up to 1e-10 while requiring identical structure, ranks and sources. All other artifacts are compared byte-for-byte. Source vocabulary is versioned and checked against artifact metadata.
- Runtime dependency audit and npm audit report no known vulnerabilities at verification time.
- Browser: all three fictional cases, a live assessment, text PDF/scanned PDF/PNG upload queue, editable result and actual PDF download verified. The public three-report PDF has four pages and contains the corrected value and educational notice.
- The revised white/teal healthcare interface is checked at desktop and 390 px mobile widths. Homepage actions, form layout, report cards and direct navigation replace the exhibition. Current screenshots use the `clinical-` prefix in `docs/screenshots/`.
- Motion controls and the system reduced-motion preference disable the minimal transitions. No sound, canvas or pinned scenes remain. Mobile layouts have no horizontal page overflow.

## Remaining product limitations

This is not clinically validated. The historical dataset has undocumented provenance, only 304 distinct patterns and strong template similarity. The model can rank an inappropriate condition for sparse or unfamiliar combinations. It cannot assess urgency, estimate personal risk, use lab values or consider unrecorded context.

The local report parser recognizes a bounded vocabulary of common analytes. Complex tables, handwriting, unusual tests, non-English reports and poor scans may be partially extracted or unsupported. OCR can omit a unit or misread a value; review flags and manual editing are essential. Empty extraction does not mean normal results. PDF export currently transliterates text to the built-in Latin font, so scientific units are transliterated (for example µL → uL and 10³ → 10^3); other unsupported Unicode characters are simplified.

Gemini generation is optional. Missing credentials and provider failure produce a clearly labeled prepared explanation. A successful fallback does not prove a live Gemini response was generated.

Public deployment verification and provider-specific limitations are recorded separately in `docs/RELEASE.md`.
