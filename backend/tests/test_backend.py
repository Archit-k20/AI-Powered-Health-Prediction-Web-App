import asyncio
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pypdf import PdfWriter
from reportlab.pdfgen.canvas import Canvas

from backend.app import app
from backend.health.model import ModelService, REGISTRY, ROOT
from backend.health.reports import parse_text, extract_sync, extract_bounded
from backend.health.schemas import AssessmentInput
from backend.health import samples
from training.train import load_patterns, normalize


@pytest.fixture(scope="module")
def service():
    return ModelService()


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def text_pdf(lines=("Hemoglobin 13.8 g/dL 12.0-16.0", "Glucose 92 mg/dL 70-99")):
    output = io.BytesIO()
    canvas = Canvas(output)
    for i, line in enumerate(lines):
        canvas.drawString(40, 760 - i * 30, line)
    canvas.save()
    return output.getvalue()


def test_training_integrity(service):
    X, y, features, hashes, raw = load_patterns()
    m = service.metadata
    assert raw == 4920 and len(y) == 304 and X.shape == (304, 131)
    assert set(features).isdisjoint({normalize(label) for label in y})
    assert set(m["train_pattern_hashes"]).isdisjoint(m["test_pattern_hashes"])
    assert set(hashes) == set(m["train_pattern_hashes"] + m["test_pattern_hashes"])
    assert m["train_count"] == 243 and m["test_count"] == 61
    assert "yellow_crust_ooze" in features
    assert (
        hashlib.sha256((ROOT / "data/dataset.csv").read_bytes()).hexdigest()
        == m["dataset_sha256"]
    )
    # Verify reported metrics against the packaged, actually evaluated estimator.
    from sklearn.metrics import accuracy_score, f1_score

    indices = [hashes.index(h) for h in m["test_pattern_hashes"]]
    predicted = service.model.predict(X[indices])
    metrics = m["evaluation"]["random_forest"]
    assert accuracy_score(y[indices], predicted) == metrics["accuracy"]
    assert f1_score(y[indices], predicted, average="macro") == metrics["macro_f1"]


def test_health_and_contract(client):
    assert client.get("/api/v1/health").json()["status"] == "ready"
    assert client.get("/api/v1/symptoms").json()["version"] == REGISTRY["version"]
    assert client.get("/api/v1/model-info").json()["unique_patterns"] == 304


@pytest.mark.parametrize(
    "symptoms", [[], ["headache"], ["Fever", "fever"], ["not a symptom", "headache"]]
)
def test_invalid_inputs(client, symptoms):
    response = client.post("/api/v1/assessments", json={"symptoms": symptoms})
    assert response.status_code == 422 and "error" in response.json()


def test_aliases_and_duplicates(client):
    response = client.post(
        "/api/v1/assessments",
        json={"symptoms": [" Fever ", "Shortness of breath", "fever"]},
    )
    assert response.status_code == 200
    assert response.json()["symptoms"] == ["high_fever", "breathlessness"]


def test_confirmed_followups_and_corrections(client):
    base = {
        "symptoms": ["cough", "fatigue"],
        "followups": [
            {"question_id": "fever_check", "answer": "Yes", "confirmed": False}
        ],
    }
    assert (
        "high_fever"
        not in client.post("/api/v1/assessments", json=base).json()["symptoms"]
    )
    base["followups"][0]["confirmed"] = True
    assert (
        "high_fever" in client.post("/api/v1/assessments", json=base).json()["symptoms"]
    )
    base["followups"][0]["answer"] = "No"
    assert (
        "high_fever"
        not in client.post("/api/v1/assessments", json=base).json()["symptoms"]
    )
    base["followups"] = [
        {"question_id": "duration", "answer": "1–3 days", "confirmed": True}
    ]
    assert client.post("/api/v1/assessments", json=base).json()["symptoms"] == [
        "cough",
        "fatigue",
    ]


def test_shap_correct_class_and_feature_positions(service):
    inputs = ["headache", "nausea", "acidity", "blurred_and_distorted_vision"]
    result = service.assess(AssessmentInput(symptoms=inputs))
    x = np.array([[int(f in inputs) for f in service.features]])
    values = service.explainer.shap_values(x)
    for match in result.matches:
        index = service.model.classes_.tolist().index(match.condition)
        expected = values[0, :, index]
        assert np.isclose(
            service.explainer.expected_value[index] + sum(expected), match.model_output
        )
        for c in match.contributions:
            assert c.contribution == expected[service.features.index(c.symptom_id)]
            assert c.present == (c.symptom_id in inputs)


def test_missing_model_and_metadata_validation(client, service, tmp_path):
    client.app.state.model = None
    assert client.get("/api/v1/health").json()["status"] == "degraded"
    assert (
        client.post(
            "/api/v1/assessments", json={"symptoms": ["headache", "nausea"]}
        ).status_code
        == 503
    )
    (tmp_path / "metadata.json").write_text(json.dumps(service.metadata))
    (tmp_path / "model.joblib").write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        ModelService(tmp_path)


def test_local_parser_preserves_missing_values():
    findings = parse_text(
        "Hemoglobin 13.8 g/dL 12.0-16.0\nGlucose pending\nTSH 2.1 mIU/L\nrandom header",
        2,
        "embedded_text",
    )
    assert len(findings) == 3
    assert findings[0].reference_range == "12.0-16.0" and findings[0].source_page == 2
    assert (
        findings[1].value is None
        and findings[1].unit is None
        and findings[1].needs_review
    )
    assert findings[2].reference_range is None


def test_pdf_text_and_partial():
    result = extract_sync(text_pdf(), "pdf")
    assert len(result["findings"]) == 2 and not result["partial"]
    result = extract_sync(
        text_pdf(["This fictional page contains no supported laboratory rows."]), "pdf"
    )
    assert result["partial"] and not result["findings"]


def test_pdf_page_and_image_limits():
    writer = PdfWriter()
    for _ in range(11):
        writer.add_blank_page(612, 792)
    out = io.BytesIO()
    writer.write(out)
    with pytest.raises(ValueError, match="ten"):
        extract_sync(out.getvalue(), "pdf")
    out = io.BytesIO()
    Image.new("L", (5000, 4001)).save(out, format="PNG")
    with pytest.raises(ValueError, match="20 megapixels"):
        extract_sync(out.getvalue(), "image")


def test_upload_limits_malformed_and_types(client):
    assert (
        client.post(
            "/api/v1/lab-reports/extract",
            files={"file": ("file.pdf", b"not a pdf", "application/pdf")},
        ).status_code
        == 415
    )
    assert (
        client.post(
            "/api/v1/lab-reports/extract",
            files={"file": ("file.png", b"x" * (10 * 1024 * 1024 + 1), "image/png")},
        ).status_code
        == 413
    )
    response = client.post(
        "/api/v1/lab-reports/extract",
        files={"file": ("file.pdf", b"%PDF-malformed", "application/pdf")},
    )
    assert response.status_code == 422


def test_live_health_data_cannot_reach_gemini(client, monkeypatch, caplog, capsys):
    forbidden = AsyncMock(side_effect=AssertionError("Network must not be called"))
    monkeypatch.setattr("httpx.AsyncClient.post", forbidden)
    response = client.post(
        "/api/v1/assessments", json={"symptoms": ["cough", "fatigue"]}
    )
    assert response.status_code == 200
    for body in [
        {"text": "PRIVATE_HEALTH_SENTINEL"},
        {"symptoms": ["cough"]},
        {"reports": "PRIVATE_HEALTH_SENTINEL"},
    ]:
        assert client.post("/api/v1/samples/cold/explain", json=body).status_code == 422
    assert (
        client.post(
            "/api/v1/samples/cold/explain?text=PRIVATE_HEALTH_SENTINEL"
        ).status_code
        == 422
    )
    assert client.post("/api/v1/samples/unknown/explain").status_code == 404
    assert not forbidden.called
    assert "PRIVATE_HEALTH_SENTINEL" not in caplog.text + capsys.readouterr().out


def test_real_report_never_calls_network(client, monkeypatch):
    forbidden = AsyncMock(side_effect=AssertionError("No network"))
    monkeypatch.setattr("httpx.AsyncClient.post", forbidden)
    response = client.post(
        "/api/v1/lab-reports/extract",
        files={"file": ("fictional.pdf", text_pdf(), "application/pdf")},
    )
    assert response.status_code == 200 and len(response.json()["findings"]) == 2
    assert not forbidden.called


@pytest.mark.parametrize("failure", ["no_key", "quota", "bad_citation", "malformed"])
def test_sample_fallbacks(monkeypatch, failure):
    if failure == "no_key":
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
        if failure == "quota":
            mock = AsyncMock(side_effect=RuntimeError("quota"))
        else:
            import httpx

            content = (
                "not json"
                if failure == "malformed"
                else json.dumps(
                    {
                        "text": "A fictional explanation long enough for schema validation.",
                        "source_ids": ["invented"],
                    }
                )
            )
            mock = AsyncMock(
                return_value=httpx.Response(
                    200,
                    json={"candidates": [{"content": {"parts": [{"text": content}]}}]},
                    request=httpx.Request("POST", "https://example.com"),
                )
            )
        monkeypatch.setattr("httpx.AsyncClient.post", mock)
    result = asyncio.run(samples.explain_sample("cold"))
    assert result.mode == "prepared" and result.source_ids == ["medlineplus-commoncold"]


def test_timeout_and_cancellation_cleanup():
    async def disconnected():
        return False

    async def gone():
        return True

    with pytest.raises(TimeoutError):
        asyncio.run(extract_bounded(text_pdf(), "pdf", disconnected, timeout=-1))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(extract_bounded(text_pdf(), "pdf", gone))


def test_cors_and_no_cache(client):
    good = client.get("/api/v1/health", headers={"Origin": "http://localhost:5173"})
    assert good.headers["access-control-allow-origin"] == "http://localhost:5173"
    bad = client.get("/api/v1/health", headers={"Origin": "https://untrusted.example"})
    assert "access-control-allow-origin" not in bad.headers
    assert good.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "filename,method",
    [
        ("fictional-text.pdf", "embedded_text"),
        ("fictional-scanned.pdf", "ocr"),
        ("fictional-image.png", "ocr"),
        ("fictional-image.jpg", "ocr"),
    ],
)
def test_real_report_fixtures(filename, method):
    import shutil

    if method == "ocr" and not shutil.which("tesseract"):
        pytest.skip(
            "Run container verification or install Tesseract for real OCR tests"
        )
    data = (Path(__file__).parent / "fixtures" / filename).read_bytes()
    result = extract_sync(data, "pdf" if filename.endswith("pdf") else "image")
    assert len(result["findings"]) == 4
    assert result["findings"][0]["value"] == "13.8"
    assert result["findings"][0]["method"] == method
    assert result["findings"][2]["test_name"] == "Platelets"
    if method == "embedded_text":
        assert result["findings"][2]["unit"] == "10^3/uL"
    else:
        assert result["findings"][2]["unit"] in (None, "10^3/uL")
        assert result["findings"][2]["needs_review"]
    assert result["findings"][3]["value"] is None
    assert result["partial"]


def test_rate_limit(client):
    for _ in range(24):
        response = client.post("/api/v1/samples/unknown/explain")
        assert response.status_code == 404
    assert client.post("/api/v1/samples/unknown/explain").status_code == 429


def test_bounded_extraction_queue(client):
    client.app.state.extract_waiting = 3
    response = client.post(
        "/api/v1/lab-reports/extract", files={"file": ("test.pdf", text_pdf())}
    )
    assert response.status_code == 429


def test_streaming_body_limit():
    from backend.health.limits import BodyLimitMiddleware

    messages = []

    async def receiver():
        return {"type": "http.request", "body": b"123456", "more_body": False}

    async def sender(message):
        messages.append(message)

    async def child(scope, receive, send):
        await receive()

    asyncio.run(
        BodyLimitMiddleware(child, maximum=5)({"type": "http"}, receiver, sender)
    )
    assert messages[0]["status"] == 413


def test_raw_health_content_not_in_errors_or_logs(client, caplog, capsys):
    sentinel = "HEALTH_PRIVATE_TEXT_83719"
    response = client.post(
        "/api/v1/assessments", json={"symptoms": [sentinel, "cough"]}
    )
    assert sentinel not in response.text
    response = client.post(
        "/api/v1/lab-reports/extract",
        files={"file": (sentinel + ".pdf", b"%PDF-" + sentinel.encode())},
    )
    assert sentinel not in response.text + caplog.text + capsys.readouterr().out


def test_sample_prompts_contain_only_allowlisted_content(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    captured = []

    async def fake(self, url, **kwargs):
        import httpx

        captured.append(kwargs["json"])
        text = json.dumps(
            {
                "text": "This fictional example illustrates a model pattern. It is not a diagnosis.",
                "source_ids": ["medlineplus-commoncold"],
            }
        )
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": text}]}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("httpx.AsyncClient.post", fake)
    response = asyncio.run(samples.explain_sample("cold"))
    assert response.mode == "gemini"
    prompt = json.loads(captured[0]["contents"][0]["parts"][0]["text"])
    assert prompt["fictional_case"] == samples.SAMPLES["cold"]
    assert (
        prompt["retrieved_catalog_passage"]["sources"][0]["id"]
        == "medlineplus-commoncold"
    )


def test_cors_wraps_rate_limit_errors(client):
    client.app.state.rates["testclient"] = __import__("collections").deque(
        [__import__("time").monotonic()] * 24
    )
    response = client.post(
        "/api/v1/samples/cold/explain", headers={"Origin": "http://localhost:5173"}
    )
    assert response.status_code == 429
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
