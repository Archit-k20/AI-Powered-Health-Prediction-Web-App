import asyncio
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
import json
import os
import shutil
import time

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from backend.health.limits import BodyLimitMiddleware
from backend.health.model import ROOT, REGISTRY, ModelService, InputError
from backend.health.reports import MAX_BYTES, extract_bounded
from backend.health.samples import SAMPLES, explain_sample
from backend.health.schemas import (
    AssessmentInput,
    Assessment,
    Extraction,
    Health,
    ModelInfo,
    Registry,
    SampleExplanation,
    ErrorResponse,
)


@asynccontextmanager
async def lifespan(app):
    try:
        app.state.model = ModelService(os.getenv("MODEL_DIR"))
    except Exception:
        app.state.model = None
    app.state.extract_lock = asyncio.Semaphore(1)
    app.state.extract_waiting = 0
    app.state.assess_lock = asyncio.Semaphore(2)
    app.state.rates = OrderedDict()
    yield


app = FastAPI(
    title="The Shape of Health",
    version="1.1.0",
    lifespan=lifespan,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
origins = [
    x.strip()
    for x in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if x.strip() and x.strip() != "*"
]
app.add_middleware(BodyLimitMiddleware)


def error(code, message, status):
    return JSONResponse(
        {"error": {"code": code, "message": message}}, status_code=status
    )


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    # FastAPI's default includes the submitted input. Deliberately omit it.
    return error(
        "invalid_input",
        "Check your selections and try again. Select at least two supported symptoms for an assessment.",
        422,
    )


@app.exception_handler(HTTPException)
async def http_error(request, exc):
    return error("request_failed", str(exc.detail), exc.status_code)


@app.middleware("http")
async def safeguards(request, call_next):
    length = request.headers.get("content-length", "0")
    if not length.isdigit() or int(length) > MAX_BYTES + 65536:
        return error("upload_too_large", "Each report must be 10 MB or smaller.", 413)
    if request.method == "POST":
        # Bounded, in-memory per-peer and service-wide quotas; no raw IP or request logging.
        now = time.monotonic()
        rates = request.app.state.rates
        peer = request.client.host if request.client else "unknown"
        for key, limit in [(peer, 24), ("__global__", 80)]:
            queue = rates.setdefault(key, deque())
            while queue and queue[0] < now - 60:
                queue.popleft()
            if len(queue) >= limit:
                return error(
                    "rate_limited",
                    "The demo is busy. Wait a minute, or explore a fictional example.",
                    429,
                )
            queue.append(now)
            rates.move_to_end(key)
        while len(rates) > 4096:
            rates.popitem(last=False)
    try:
        response = await call_next(request)
    except Exception:
        # Do not let framework exception logging expose response/input validation data.
        return error(
            "service_error",
            "The service could not complete this request. Please try again.",
            500,
        )
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.get("/api/v1/health", response_model=Health)
async def health(request: Request):
    model = request.app.state.model
    readiness = Health(
        status="ready" if model else "degraded",
        model_version=model.metadata["version"] if model else None,
        capabilities={
            "assessments": bool(model),
            "pdf_text": True,
            "ocr": bool(shutil.which("tesseract")),
            "report_review": True,
            "fictional_explanations": True,
            "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
        },
    )
    if not model:
        return JSONResponse(readiness.model_dump(), status_code=503)
    return readiness


@app.get("/api/v1/symptoms", response_model=Registry)
async def symptoms():
    return REGISTRY


@app.get("/api/v1/model-info", response_model=ModelInfo)
async def model_info():
    return json.loads((ROOT / "shared/model-info.json").read_text())


@app.post("/api/v1/assessments", response_model=Assessment)
async def assess(payload: AssessmentInput, request: Request):
    service = request.app.state.model
    if service is None:
        return error(
            "model_unavailable",
            "The model is not ready. Try again shortly or open a fictional walkthrough.",
            503,
        )
    try:
        async with asyncio.timeout(15):
            async with request.app.state.assess_lock:
                return await run_in_threadpool(service.assess, payload)
    except InputError as exc:
        return error("unsupported_input", str(exc), 422)
    except TimeoutError:
        return error(
            "assessment_timeout", "The model took too long. Try again shortly.", 504
        )


@app.post(
    "/api/v1/lab-reports/extract", response_model=Extraction, include_in_schema=False
)
@app.post("/api/v1/reports/extract", response_model=Extraction)
async def extract(request: Request, file: UploadFile = File(...)):
    data = await file.read(MAX_BYTES + 1)
    await file.close()
    if len(data) > MAX_BYTES:
        return error("upload_too_large", "Each report must be 10 MB or smaller.", 413)
    if data.startswith(b"%PDF-"):
        kind = "pdf"
    elif data.startswith(b"\x89PNG\r\n\x1a\n") or data.startswith(b"\xff\xd8\xff"):
        kind = "image"
    else:
        return error("unsupported_file", "Choose a valid PDF, JPG or PNG file.", 415)
    if request.app.state.extract_waiting >= 3:
        return error(
            "extraction_busy",
            "The report reader is busy. Please try again shortly.",
            429,
        )
    request.app.state.extract_waiting += 1
    try:
        async with asyncio.timeout(105):
            async with request.app.state.extract_lock:
                return await extract_bounded(data, kind, request.is_disconnected)
    except TimeoutError:
        return error(
            "extraction_timeout",
            "Extraction timed out. Try a shorter PDF or a clearer image.",
            504,
        )
    except ValueError as exc:
        return error("extraction_failed", str(exc), 422)
    finally:
        request.app.state.extract_waiting -= 1


@app.post("/api/v1/samples/{sample_id}/explain", response_model=SampleExplanation)
async def sample_explanation(sample_id: str, request: Request):
    if sample_id not in SAMPLES:
        return error(
            "unknown_sample", "Choose one of the three fictional examples.", 404
        )
    if request.query_params or await request.body():
        return error(
            "sample_input_forbidden",
            "This endpoint accepts predefined fictional case IDs only; no text, health inputs or uploads.",
            422,
        )
    return await explain_sample(sample_id)


# CORS must wrap error/rate-limit responses as well as successful routes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
