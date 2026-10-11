# Verification record

Local verification on 2026-10-11:

- 32 backend tests pass inside the Linux runtime with Tesseract. Native macOS runs skip the three OCR cases when Tesseract is absent.
- Thirteen frontend tests pass, covering all offline examples, aliases, duplicate removal, follow-up confirmation and corrections, backend failure, editable reports, sequential three-file processing, partial failure, keyboard tabs, memory-only data, PDF pagination, chapter controls and a WebGL failure fallback.
- Production TypeScript and Vite build passes. Heavy Three.js, workspace and PDF code load separately.
- Retraining reproduces model checksum, split hashes and measured metrics. Source vocabulary is versioned and checked against artifact metadata.
- Runtime dependency audit and npm audit report no known vulnerabilities at verification time.
- Browser: all three fictional cases, a live assessment, text PDF/scanned PDF/PNG upload queue, editable result and actual PDF download verified. The public three-report PDF has four pages and contains the corrected value and educational notice.
- Visual comparison against The Shape of Intelligence at desktop and 390 px mobile widths. The health title, original sculpture and product copy differ; pale/charcoal scenes, large typography, chapter transitions and bottom controls follow the reference. Screenshots are in `docs/screenshots/`.
- Motion can be disabled without losing navigation or content. Mobile chapter selection moves to the correct section with no horizontal page overflow. Sound starts off and requires an explicit switch.

## Remaining product limitations

This is not clinically validated. The historical dataset has undocumented provenance, only 304 distinct patterns and strong template similarity. The model can rank an inappropriate condition for sparse or unfamiliar combinations. It cannot assess urgency, estimate personal risk, use lab values or consider unrecorded context.

The local report parser recognizes a bounded vocabulary of common analytes. Complex tables, handwriting, unusual tests, non-English reports and poor scans may be partially extracted or unsupported. OCR can omit a unit or misread a value; review flags and manual editing are essential. Empty extraction does not mean normal results. PDF export currently transliterates text to the built-in Latin font, so scientific units are transliterated (for example µL → uL and 10³ → 10^3); other unsupported Unicode characters are simplified.

Gemini generation is optional. Missing credentials and provider failure produce a clearly labeled prepared explanation. A successful fallback does not prove a live Gemini response was generated.

Public deployment verification and provider-specific limitations are recorded separately in `docs/RELEASE.md`.
