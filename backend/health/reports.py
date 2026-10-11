"""Local extraction only. Files and their text never leave this process tree."""

import asyncio
import io
import logging
import multiprocessing as mp
import os
import re
import shutil
import signal
import time

from PIL import Image
from pypdf import PdfReader
import pypdfium2 as pdfium
import pytesseract

from .schemas import Extraction, Finding

# Library diagnostics may include fragments of malformed input. Suppress them.
_pdf_log = logging.getLogger("pypdf")
_pdf_log.addHandler(logging.NullHandler())
_pdf_log.propagate = False

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
TESTS = {
    "Hemoglobin": r"(?:ha?emoglobin|hgb|hb)(?!a1c)",
    "White blood cells": r"(?:white blood (?:cells?|count)|wbc|total leucocyte count)",
    "Platelets": r"(?:platelets?(?: count)?|plt)",
    "Glucose": r"(?:(?:fasting |random )?(?:blood )?glucose|blood sugar)",
    "HbA1c": r"(?:hba1c|glycated ha?emoglobin)",
    "Creatinine": r"(?:serum )?creatinine",
    "Total cholesterol": r"(?:total )?cholesterol",
    "HDL cholesterol": r"hdl(?: cholesterol)?",
    "LDL cholesterol": r"ldl(?: cholesterol)?",
    "Triglycerides": r"triglycerides?",
    "TSH": r"(?:tsh|thyroid stimulating hormone)",
    "Vitamin D": r"(?:vitamin d|25[- ]oh vitamin d)",
    "Vitamin B12": r"vitamin b12",
    "Bilirubin": r"(?:total )?bilirubin",
    "ALT": r"(?:alt|sgpt)",
    "AST": r"(?:ast|sgot)",
    "Urea": r"(?:blood )?urea",
    "Sodium": r"sodium",
    "Potassium": r"potassium",
    "Calcium": r"(?:serum )?calcium",
    "Hematocrit": r"(?:ha?ematocrit|hct|pcv)",
    "Red blood cells": r"(?:red blood (?:cells?|count)|rbc)",
}
NUM = r"[<>≤≥]?\s*\d+(?:[.,]\d+)?"
RANGE = re.compile(
    r"(?:\b(?:ref(?:erence)?(?: range)?|range)\s*[:=]?\s*|[([]\s*)?("
    + NUM
    + r"\s*[-–]\s*"
    + NUM
    + r")"
)
UNIT = re.compile(
    r"^\s*((?:[x×]\s*)?10(?:\^?\d+|[³⁶⁹])\s*/\s*[a-zA-Zµμ]+|(?:[a-zA-Zµμ%]+(?:\^?\d+)?)(?:\s*/\s*[a-zA-Zµμ]+(?:\^?\d+)?)?)"
)


def parse_text(text, page, method):
    findings = []
    for line in text.splitlines():
        line = " ".join(line.split())
        for name, pattern in TESTS.items():
            matched = re.match(r"^" + pattern + r"\b\s*[:|\-]?\s*", line, re.I)
            if not matched:
                continue
            tail = line[matched.end() :]
            number = re.match(NUM, tail)
            value = number.group(0).strip() if number else None
            after = tail[number.end() :].strip() if number else tail
            ref = RANGE.search(after)
            reference = ref.group(1).strip() if ref else None
            unit_match = UNIT.match(after)
            unit = unit_match.group(1).strip() if unit_match else None
            if unit and unit.lower() in (
                "ref",
                "reference",
                "range",
                "low",
                "high",
                "normal",
                "pending",
            ):
                unit = None
            findings.append(
                Finding(
                    test_name=name,
                    value=value,
                    unit=unit,
                    reference_range=reference,
                    source_page=page,
                    method=method,
                    needs_review=method == "ocr" or not all([value, unit, reference]),
                )
            )
            break
    return findings


def ocr(image):
    if image.width * image.height > MAX_PIXELS:
        raise ValueError("Image exceeds 20 megapixels.")
    if not shutil.which("tesseract"):
        raise RuntimeError("OCR is unavailable on this server.")
    return pytesseract.image_to_string(image, config="--psm 6", timeout=18)


def extract_sync(data, kind):
    findings, warnings = [], []
    if kind == "pdf":
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise ValueError(
                "Password-protected PDFs are not supported. Use an unlocked copy."
            )
        pages = len(reader.pages)
        if pages > 10:
            raise ValueError("Reports are limited to ten PDF pages.")
        if not pages:
            raise ValueError("This PDF has no pages.")
        document = None
        try:
            for index, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                    method = "embedded_text"
                    if len(text.strip()) < 30:
                        document = document or pdfium.PdfDocument(data)
                        pdf_page = document[index]
                        try:
                            w, h = pdf_page.get_size()
                            if w * h * 4 > MAX_PIXELS:
                                raise ValueError("Rendered page exceeds 20 megapixels.")
                            bitmap = pdf_page.render(scale=2)
                            try:
                                text = ocr(bitmap.to_pil())
                            finally:
                                bitmap.close()
                        finally:
                            pdf_page.close()
                        method = "ocr"
                    found = parse_text(text, index + 1, method)
                    findings.extend(found)
                    if not found:
                        warnings.append(
                            f"Page {index + 1}: no supported test rows recognized. Check the original; this does not mean normal results."
                        )
                except Exception:
                    warnings.append(
                        f"Page {index + 1}: could not extract this page. Review the original or upload a clearer image."
                    )
        finally:
            if document:
                document.close()
    else:
        pages = 1
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ("JPEG", "PNG"):
                raise ValueError("Only PDF, JPG and PNG reports are supported.")
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("Image exceeds 20 megapixels.")
            text = ocr(image.convert("RGB"))
            findings = parse_text(text, 1, "ocr")
        if not findings:
            warnings.append(
                "No supported test rows recognized. Check the original report."
            )
    if any(f.needs_review for f in findings):
        warnings.append(
            "Some fields are uncertain or missing. Correct them before exporting."
        )
    return Extraction(
        findings=findings, warnings=warnings, pages=pages, partial=bool(warnings)
    ).model_dump()


def worker(connection, data, kind):
    if hasattr(os, "setsid"):
        os.setsid()
    try:
        connection.send(("ok", extract_sync(data, kind)))
    except ValueError as exc:
        connection.send(("invalid", str(exc)))
    except Exception:
        connection.send(
            (
                "failed",
                "This report could not be read. Try an unlocked PDF or a clearer JPG or PNG.",
            )
        )
    finally:
        connection.close()


async def extract_bounded(data, kind, disconnected, timeout=90):
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=worker, args=(child, data, kind), daemon=True)
    process.start()
    child.close()
    started = time.monotonic()
    try:
        while not parent.poll():
            if await disconnected():
                raise asyncio.CancelledError()
            if time.monotonic() - started > timeout:
                raise TimeoutError(
                    "Extraction timed out. Try fewer pages or a clearer image."
                )
            if not process.is_alive():
                raise ValueError("The extraction worker stopped. Try another report.")
            await asyncio.sleep(0.08)
        status, result = parent.recv()
        if status != "ok":
            raise ValueError(result)
        return Extraction.model_validate(result)
    finally:
        if process.is_alive():
            try:
                if hasattr(os, "killpg") and os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            except ProcessLookupError:
                pass
        await asyncio.to_thread(process.join, 2)
        if process.is_alive():
            process.kill()
        parent.close()
