"""Local document extraction, with bounded processes and no external AI calls."""

import asyncio
import io
import logging
import multiprocessing as mp
import os
import signal
import time

from PIL import Image
from pypdf import PdfReader
import pypdfium2 as pdfium

from .document_layout import MAX_PIXELS, pdf_lines, ocr_lines, readable
from .lab_parser import TESTS, parse_text, parse_lines
from .report_clinical import classify_document, extract_mentions
from .schemas import Extraction, ReportPage

_pdf_log = logging.getLogger("pypdf")
_pdf_log.addHandler(logging.NullHandler())
_pdf_log.propagate = False
MAX_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def read_page(lines, page, method, warnings=None):
    warnings = list(warnings or [])
    text = "\n".join(line.text for line in lines)
    supported = readable(text)
    findings = parse_lines(lines, page, method) if supported else []
    kind = classify_document(text, bool(findings)) if supported else "unknown"
    mentions = extract_mentions(lines, page, method, kind) if supported else []
    if any("recognition is uncertain" in warning for warning in warnings):
        for mention in mentions:
            mention.eligible = False
            mention.review_reason = "The scan is too uncertain for symptom transfer. Check the original and enter current symptoms manually."
    if any("rotated" in warning for warning in warnings):
        # Layout coordinates refer to the OCR orientation, not the original preview.
        for item in [*findings, *mentions]:
            if item.evidence:
                item.evidence.bbox = None
    if kind == "template":
        findings = []
        warnings.append(
            "This appears to be an unfilled report template. No patient findings were identified. Upload a completed report or enter symptoms yourself."
        )
    elif not supported:
        warnings.append(
            "This page could not be reliably read as English text. Upload a clearer English report or enter the information manually."
        )
    elif not findings and not mentions:
        warnings.append(
            "No supported clinical findings were identified on this page. This does not mean the report is normal; review the original or enter symptoms yourself."
        )
    if len(findings) >= 200 or len(mentions) >= 100:
        warnings.append(
            "This page reached the extraction limit. Review the remaining content manually."
        )
    if method == "ocr":
        for finding in findings:
            finding.needs_review = True
    if any(f.needs_review for f in findings):
        warnings.append(
            "Some lab fields need review. Missing values are deliberately left blank."
        )
    if len(text) > 20000:
        warnings.append(
            "The reading transcript was shortened; use the original for the remaining text."
        )
    return (
        findings,
        mentions,
        ReportPage(
            page=page,
            document_type=kind,
            method=method,
            quality="unreadable"
            if not supported
            else "review"
            if warnings
            else "readable",
            text=text[:20000],
            warnings=warnings,
        ),
    )


def extract_sync(data, kind):
    findings, mentions, page_details = [], [], []

    def collect(lines, number, method, warnings=None):
        found, clinical, detail = read_page(lines, number, method, warnings)
        findings.extend(found)
        mentions.extend(clinical)
        page_details.append(detail)

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
        document = pdfium.PdfDocument(data)
        try:
            for index in range(pages):
                page = document[index]
                method, warnings = "embedded_text", []
                try:
                    lines = pdf_lines(page)
                    text = "\n".join(line.text for line in lines)
                    # A selectable header must not hide an image-only report body.
                    images = list(
                        page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE])
                    )
                    width, height = page.get_size()
                    substantial_image = any(
                        (obj.get_bounds()[2] - obj.get_bounds()[0])
                        * (obj.get_bounds()[3] - obj.get_bounds()[1])
                        > width * height * 0.25
                        for obj in images
                    )
                    if not readable(text) or substantial_image:
                        scale = min(
                            max(2.5, 1800 / width),
                            (MAX_PIXELS / (width * height)) ** 0.5,
                        )
                        if width * height > MAX_PIXELS or scale < 1:
                            raise ValueError("Rendered page exceeds 20 megapixels.")
                        bitmap = page.render(scale=scale)
                        try:
                            scanned, ocr_warnings = ocr_lines(bitmap.to_pil())
                        finally:
                            bitmap.close()
                        if scanned and (
                            not readable(text)
                            or sum(len(l.text) for l in scanned) >= len(text)
                        ):
                            lines, method = scanned, "ocr"
                            warnings.extend(ocr_warnings)
                        elif substantial_image:
                            warnings.append(
                                "An image region may contain unreadable information. Check the original page."
                            )
                    collect(lines, index + 1, method, warnings)
                except Exception:
                    page_details.append(
                        ReportPage(
                            page=index + 1,
                            document_type="unknown",
                            method=method,
                            quality="unreadable",
                            text="",
                            warnings=[
                                "Could not read this page. Try a clearer image or enter the information manually."
                            ],
                        )
                    )
                finally:
                    page.close()
        finally:
            document.close()
    else:
        pages = 1
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ("JPEG", "PNG"):
                raise ValueError("Only PDF, JPG and PNG reports are supported.")
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("Image exceeds 20 megapixels.")
            lines, warnings = ocr_lines(image)
            collect(lines, 1, "ocr", warnings)
    warnings = [f"Page {p.page}: {w}" for p in page_details for w in p.warnings]
    # Conflicting mentions are evidence for review, not a vote to add a symptom.
    for mention in mentions:
        if mention.eligible and any(
            other.symptom_id == mention.symptom_id
            and other.id != mention.id
            and (
                other.polarity != "affirmed"
                or other.timing != "unspecified"
                or other.subject != "patient"
            )
            for other in mentions
        ):
            mention.eligible = False
            mention.review_reason = "The report contains conflicting or differently timed mentions. Clarify in Symptoms; this item cannot be transferred automatically."
    return Extraction(
        findings=findings,
        mentions=mentions,
        page_details=page_details,
        warnings=warnings,
        pages=pages,
        partial=bool(warnings),
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
