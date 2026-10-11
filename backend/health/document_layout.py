"""Bounded page-local layout extraction. No external services or persistence."""

from dataclasses import dataclass
import re
import shutil
from statistics import median

from PIL import Image, ImageOps
import pytesseract

MAX_PIXELS = 20_000_000
MAX_CHARS = 60_000


@dataclass
class Word:
    text: str
    x: float
    y: float
    right: float
    bottom: float
    confidence: float = 100


@dataclass
class Line:
    words: list[Word]

    @property
    def text(self):
        return " ".join(w.text for w in self.words)

    @property
    def bbox(self):
        x, y = min(w.x for w in self.words), min(w.y for w in self.words)
        return [
            round(x, 5),
            round(y, 5),
            round(max(w.right for w in self.words) - x, 5),
            round(max(w.bottom for w in self.words) - y, 5),
        ]


def group_lines(words):
    """Use physical baselines, independent of PDF content-stream reading order."""
    rows = []
    for word in sorted(words, key=lambda w: ((w.y + w.bottom) / 2, w.x)):
        center = (word.y + word.bottom) / 2
        if rows and abs(center - median((w.y + w.bottom) / 2 for w in rows[-1])) < max(
            0.003, (word.bottom - word.y) * 0.55
        ):
            rows[-1].append(word)
        else:
            rows.append([word])
    return [Line(sorted(row, key=lambda w: w.x)) for row in rows]


def pdf_lines(page):
    width, height = page.get_size()
    textpage = page.get_textpage()
    words, chars = [], []

    def flush():
        if chars:
            words.append(
                Word(
                    "".join(c[0] for c in chars),
                    min(c[1] for c in chars),
                    min(c[2] for c in chars),
                    max(c[3] for c in chars),
                    max(c[4] for c in chars),
                )
            )
            chars.clear()

    try:
        count = textpage.count_chars()
        if count > MAX_CHARS:
            raise ValueError("Page text exceeds the reading limit.")
        for index in range(count):
            char = textpage.get_text_range(index, 1)
            if not char or char.isspace():
                flush()
                continue
            left, bottom, right, top = textpage.get_charbox(index)
            if chars and (
                abs((height - top) / height - chars[-1][2]) > 0.01
                or left / width - chars[-1][3] > 0.012
            ):
                flush()
            chars.append(
                (
                    char,
                    max(0, left / width),
                    max(0, (height - top) / height),
                    min(1, right / width),
                    min(1, (height - bottom) / height),
                )
            )
        flush()
        return group_lines(words)
    finally:
        textpage.close()


def ocr_lines(image):
    if image.width * image.height > MAX_PIXELS:
        raise ValueError("Image exceeds 20 megapixels.")
    if not shutil.which("tesseract"):
        raise RuntimeError("OCR is unavailable on this server.")
    image = ImageOps.exif_transpose(image)
    if image.width < 1800:
        scale = min(
            1800 / image.width, (MAX_PIXELS / (image.width * image.height)) ** 0.5
        )
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    warnings = []
    # OSD may fail on sparse pages. Never treat failure as an orientation verdict.
    try:
        osd = pytesseract.image_to_osd(
            image, output_type=pytesseract.Output.DICT, timeout=5
        )
        rotation = int(osd.get("rotate", 0))
        if rotation and float(osd.get("orientation_conf", 0)) >= 5:
            image = image.rotate(-rotation, expand=True)
            warnings.append(
                "The page was rotated for reading; compare the extracted text with the original."
            )
    except (pytesseract.TesseractError, RuntimeError):
        pass
    image = ImageOps.autocontrast(image.convert("L"))
    data = pytesseract.image_to_data(
        image,
        lang="eng",
        config="--psm 3",
        output_type=pytesseract.Output.DICT,
        timeout=18,
    )
    confidences = [
        float(c) for c, text in zip(data["conf"], data["text"]) if text.strip()
    ]
    if confidences and median(confidences) < 65:
        # Retry a dense text-block layout, without assuming either pass is correct.
        try:
            alternate = pytesseract.image_to_data(
                image,
                lang="eng",
                config="--psm 6",
                output_type=pytesseract.Output.DICT,
                timeout=18,
            )
            alternate_conf = [
                float(c)
                for c, text in zip(alternate["conf"], alternate["text"])
                if text.strip()
            ]
            if len(alternate_conf) >= len(confidences) * 0.7 and median(
                alternate_conf
            ) > median(confidences):
                data = alternate
        except (pytesseract.TesseractError, RuntimeError):
            pass
    words = []
    for i, value in enumerate(data["text"]):
        value = value.strip()
        if not value:
            continue
        x, y, w, h = (data[k][i] for k in ("left", "top", "width", "height"))
        words.append(
            Word(
                value,
                x / image.width,
                y / image.height,
                (x + w) / image.width,
                (y + h) / image.height,
                float(data["conf"][i]),
            )
        )
    if sum(len(w.text) for w in words) > MAX_CHARS:
        raise ValueError("Page text exceeds the reading limit.")
    if words and median(w.confidence for w in words) < 65:
        warnings.append(
            "Text recognition is uncertain. Use a clearer scan and check every extracted item."
        )
    warnings.append(
        "OCR was used. Handwriting and non-English text are not reliably supported; review the original."
    )
    return group_lines(words), warnings


def text_lines(text):
    # Used for plain-text parsing tests; position is deliberately not fabricated.
    return [
        Line([Word(line.strip(), 0, 0, 0, 0)])
        for line in text.splitlines()
        if line.strip()
    ]


def readable(text):
    letters = re.findall(r"[^\W\d_]", text, re.UNICODE)
    ascii_letters = sum(c.isascii() for c in letters)
    return (
        len(letters) >= 12
        and ascii_letters / max(1, len(letters)) >= 0.85
        and text.count("\ufffd") < max(2, len(text) * 0.01)
    )
