"""Fictional regression cases; no user reports are stored in the repository."""

import io
import json
import re

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas
from PIL import Image

from backend.app import app
from backend.health.document_layout import text_lines
from backend.health.lab_parser import parse_text
from backend.health.report_clinical import classify_document, extract_mentions
from backend.health.reports import extract_sync, read_page


def clinical(text, kind="clinical_note"):
    return extract_mentions(text_lines(text), 1, "embedded_text", kind)


def pdf_rows(rows):
    out = io.BytesIO()
    pdf = canvas.Canvas(out)
    for x, y, text in rows:
        pdf.drawString(x, y, text)
    pdf.save()
    return out.getvalue()


@pytest.mark.parametrize(
    "text,value,unit,reference",
    [
        ("1. Hemoglobin 13.8 g/dL 12-16", "13.8", "g/dL", "12-16"),
        ("Glucose 70-99 92 mg/dL", "92", "mg/dL", "70-99"),
        ("Glucose is elevated", None, None, None),
        ("Glucose pending", None, None, None),
        ("HIV antibody: Non-reactive", "Non-reactive", None, None),
        ("TSH <0.01 mIU/L 0.4-4.0", "<0.01", "mIU/L", "0.4-4.0"),
        ("Platelets 250 10^3/uL 150-450", "250", "10^3/uL", "150-450"),
    ],
)
def test_lab_values_not_ranges_or_prose(text, value, unit, reference):
    f = parse_text(text, 2, "embedded_text")[0]
    assert (f.value, f.unit, f.reference_range) == (value, unit, reference)
    assert f.evidence.quote == text and f.evidence.page == 2


def test_columns_read_by_position_not_content_stream_order():
    # Intentionally draw in column order, reference before result.
    data = pdf_rows(
        [
            (45, 780, "Test"),
            (270, 780, "Reference range"),
            (420, 780, "Result"),
            (500, 780, "Unit"),
            (45, 750, "Hemoglobin"),
            (45, 720, "Glucose"),
            (270, 750, "12-16"),
            (270, 720, "70-99"),
            (420, 750, "13.8"),
            (420, 720, "92"),
            (500, 750, "g/dL"),
            (500, 720, "mg/dL"),
        ]
    )
    result = extract_sync(data, "pdf")
    assert [(f["value"], f["reference_range"]) for f in result["findings"]] == [
        ("13.8", "12-16"),
        ("92", "70-99"),
    ]
    assert all(f["evidence"]["bbox"] for f in result["findings"])


@pytest.mark.parametrize(
    "text,polarity,timing,subject,eligible",
    [
        ("Patient reports cough.", "affirmed", "unspecified", "patient", True),
        ("Patient denies cough.", "negated", "unspecified", "patient", False),
        ("No evidence of cough.", "negated", "unspecified", "patient", False),
        ("Cough absent.", "negated", "unspecified", "patient", False),
        ("Possible cough.", "uncertain", "unspecified", "patient", False),
        ("Previous cough.", "affirmed", "historical", "patient", False),
        ("Cough has resolved.", "affirmed", "resolved", "patient", False),
        ("Mother has cough.", "affirmed", "unspecified", "family", False),
        ("Return if cough develops.", "affirmed", "hypothetical", "patient", False),
        ("Do you have cough?", "affirmed", "hypothetical", "patient", False),
        ("No nausea but cough present.", "affirmed", "unspecified", "patient", True),
    ],
)
def test_symptom_context(text, polarity, timing, subject, eligible):
    m = next(m for m in clinical(text) if m.symptom_id == "cough")
    assert (m.polarity, m.timing, m.subject, m.eligible) == (
        polarity,
        timing,
        subject,
        eligible,
    )
    assert m.evidence.quote == text


def test_history_heading_carries_to_following_line():
    result = clinical("Family history:\nCough and fatigue.\nSymptoms: nausea.")
    assert all(
        m.subject == "family" and not m.eligible
        for m in result
        if m.symptom_id in ("cough", "fatigue")
    )
    assert next(m for m in result if m.symptom_id == "nausea").eligible


def test_lab_results_and_diagnoses_do_not_invent_symptoms():
    result = clinical(
        "Glucose 190 mg/dL 70-99\nDiagnosis: Diabetes\nPlan: watch for nausea and fatigue."
    )
    assert not any(m.eligible for m in result)
    assert any(m.category == "diagnosis" and "Diabetes" in m.label for m in result)


def test_causes_are_attributed_not_inferred():
    result = clinical("Patient reports cough due to dust exposure.")
    assert any(
        m.category == "cause"
        and m.evidence.quote == "Patient reports cough due to dust exposure."
        for m in result
    )
    assert not any(m.category == "cause" for m in clinical("Patient reports cough."))


def test_ambiguous_fever_and_unsupported_complaint_stay_unmapped():
    result = clinical("Symptoms: fever.\nSymptoms: persistent hiccups.")
    assert len(result) == 2
    assert all(m.symptom_id is None and not m.eligible for m in result)


def test_conflicting_mentions_do_not_transfer():
    result = extract_sync(
        pdf_rows(
            [
                (45, 780, "Clinical note"),
                (45, 755, "Patient reports cough."),
                (45, 730, "Patient denies cough."),
            ]
        ),
        "pdf",
    )
    assert len(result["mentions"]) == 2
    assert not any(m["eligible"] for m in result["mentions"])


def test_template_and_foreign_text_abstain():
    template = "Medical report\nReport prepared for\nFull name\nWho requested medical evaluation\nSources of information\nPresenting complaint"
    assert classify_document(template) == "template"
    f, m, p = read_page(text_lines(template), 1, "embedded_text")
    assert not f and not m and p.document_type == "template"
    assert any("unfilled" in w for w in p.warnings)
    f, m, p = read_page(
        text_lines("患者的病史及其临床症状需要进一步检查。"), 1, "embedded_text"
    )
    assert not f and not m and p.quality == "unreadable"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("DISCHARGE SUMMARY\nPatient reports cough.", "discharge_summary"),
        (
            "MRI REPORT\nClinical history: headache\nImpression: No acute abnormality.",
            "radiology",
        ),
        ("Consultation\nPatient reports fatigue.", "clinical_note"),
    ],
)
def test_document_types(text, expected):
    assert classify_document(text) == expected


def test_report_confirmation_is_enforced_by_api():
    with TestClient(app) as client:
        blocked = client.post(
            "/api/v1/assessments",
            json={
                "symptoms": ["fatigue"],
                "report_symptoms": [
                    {"symptom_id": "cough", "confirmed_current": False}
                ],
            },
        )
        assert blocked.status_code == 422
        allowed = client.post(
            "/api/v1/assessments",
            json={
                "report_symptoms": [
                    {"symptom_id": "cough", "confirmed_current": True},
                    {"symptom_id": "fatigue", "confirmed_current": True},
                ]
            },
        )
        assert allowed.status_code == 200
        assert set(allowed.json()["symptoms"]) == {"cough", "fatigue"}
        assert (
            client.post(
                "/api/v1/assessments",
                json={
                    "report_symptoms": [
                        {"symptom_id": "glucose", "confirmed_current": True}
                    ]
                },
            ).status_code
            == 422
        )


def test_new_report_endpoint_and_no_health_logging(caplog, capsys):
    sentinel = "PRIVATE_DOCUMENT_MARKER"
    with TestClient(app) as client:
        data = pdf_rows(
            [(45, 780, sentinel), (45, 750, "Patient reports cough and fatigue.")]
        )
        result = client.post(
            "/api/v1/reports/extract",
            files={"file": ("document.pdf", data, "application/pdf")},
        )
        assert result.status_code == 200
        assert {m["symptom_id"] for m in result.json()["mentions"]} == {
            "cough",
            "fatigue",
        }
        assert result.headers["cache-control"] == "no-store"
    assert sentinel not in caplog.text + capsys.readouterr().out


def test_mixed_pdf_header_and_scanned_body_use_ocr():
    import shutil
    import pypdfium2 as pdfium
    from reportlab.lib.utils import ImageReader

    if not shutil.which("tesseract"):
        pytest.skip("Container verification runs actual OCR")
    original = pdf_rows(
        [
            (45, 750, "Clinical note: fictional verification only"),
            (45, 715, "Patient reports cough and fatigue."),
            (45, 680, "Patient denies shortness of breath."),
        ]
    )
    source = pdfium.PdfDocument(original)
    page = source[0]
    bitmap = page.render(scale=2.5)
    image = bitmap.to_pil().copy()
    bitmap.close()
    page.close()
    source.close()
    out = io.BytesIO()
    pdf = canvas.Canvas(out)
    pdf.drawString(45, 810, "Selectable header with more than thirty characters")
    pdf.drawImage(ImageReader(image), 0, 0, width=595, height=780)
    pdf.save()
    result = extract_sync(out.getvalue(), "pdf")
    assert result["page_details"][0]["method"] == "ocr"
    assert {"cough", "fatigue"} <= {
        m["symptom_id"] for m in result["mentions"] if m["eligible"]
    }
    assert not any(
        m["eligible"] for m in result["mentions"] if m["symptom_id"] == "breathlessness"
    )


def test_table_header_reference_values_does_not_replace_result_column():
    data = pdf_rows(
        [
            (45, 780, "Test"),
            (230, 780, "Reference"),
            (295, 780, "values"),
            (420, 780, "Result"),
            (500, 780, "Unit"),
            (45, 750, "Glucose"),
            (230, 750, "70-99"),
            (420, 750, "92"),
            (500, 750, "mg/dL"),
        ]
    )
    result = extract_sync(data, "pdf")
    assert result["findings"][0]["value"] == "92"
    assert result["findings"][0]["reference_range"] == "70-99"


def test_noisy_template_headings_do_not_generate_clinical_inputs():
    text = "MEDICAL REPORT\nReport prepared far\nSite and time (record information for each event)\nSources of Infonmation\nPresenting comptaint"
    assert classify_document(text) == "template"
    assert not clinical(text, "template")


@pytest.mark.parametrize(
    "text",
    [
        "Patient does not have cough.",
        "Patient doesn't report cough.",
        "Patient possibly has cough.",
        "Cough cannot be excluded.",
        "Common symptoms include cough.",
    ],
)
def test_additional_nonaffirmative_context_never_transfers(text):
    assert not any(m.eligible for m in clinical(text))


def test_poor_ocr_cannot_offer_automatic_symptom_transfer():
    _, mentions, _ = read_page(
        text_lines("Patient reports cough."),
        1,
        "ocr",
        ["Text recognition is uncertain."],
    )
    assert mentions and not any(m.eligible for m in mentions)
