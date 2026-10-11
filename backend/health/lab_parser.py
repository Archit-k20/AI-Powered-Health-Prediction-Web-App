"""Extract laboratory rows with column evidence; never infer symptoms from values."""

import re
from .document_layout import text_lines
from .schemas import Evidence, Finding

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
TESTS.update(
    {
        "CRP": r"(?:c[- ]reactive protein|crp)",
        "ESR": r"(?:erythrocyte sedimentation rate|esr)",
        "Uric acid": r"(?:serum )?uric acid",
        "Albumin": r"(?:serum )?albumin",
        "Total protein": r"(?:serum )?total protein",
        "Alkaline phosphatase": r"(?:alkaline phosphatase|alp)",
        "GGT": r"(?:ggt|gamma glutamyl transferase)",
        "Direct bilirubin": r"direct bilirubin",
        "Indirect bilirubin": r"indirect bilirubin",
        "BUN": r"(?:blood urea nitrogen|bun)",
        "eGFR": r"(?:egfr|estimated glomerular filtration rate)",
        "Chloride": r"chloride",
        "Magnesium": r"magnesium",
        "Phosphorus": r"(?:phosphorus|phosphate)",
        "Free T4": r"(?:free t4|ft4)",
        "Free T3": r"(?:free t3|ft3)",
        "T4": r"(?:total t4|t4|thyroxine)",
        "T3": r"(?:total t3|t3|triiodothyronine)",
        "Ferritin": r"(?:serum )?ferritin",
        "Iron": r"(?:serum )?iron",
        "MCV": r"(?:mcv|mean corpuscular volume)",
        "MCHC": r"(?:mchc|mean corpuscular ha?emoglobin concentration)",
        "MCH": r"(?:mch|mean corpuscular ha?emoglobin)",
        "RDW": r"(?:rdw(?:[- ]cv)?|red cell distribution width)",
        "Neutrophils": r"neutrophils?",
        "Lymphocytes": r"lymphocytes?",
        "Monocytes": r"monocytes?",
        "Eosinophils": r"eosinophils?",
        "Basophils": r"basophils?",
        "Prothrombin time": r"(?:prothrombin time|pt)",
        "INR": r"(?:inr|international normalized ratio)",
        "APTT": r"(?:aptt|activated partial thromboplastin time)",
        "D-dimer": r"d[- ]dimer",
        "HIV antibody": r"hiv(?:[- ](?:1[/& ]*2|1|2))?(?: antibody| antibodies| ag/ab)?",
        "HBsAg": r"(?:hbsag|hepatitis b surface antigen)",
        "HCV antibody": r"(?:anti[- ]hcv|hcv antibody)",
        "Urine protein": r"urine protein",
        "Urine glucose": r"urine glucose",
        "Ketones": r"(?:urine )?ketones?",
    }
)
# Longest names first prevents e.g. MCH from consuming MCHC descriptions.
PATTERNS = [
    (name, re.compile(r"^" + pattern + r"\b\s*[:|\-]?\s*", re.I))
    for name, pattern in sorted(TESTS.items(), key=lambda item: -len(item[1]))
]
NUM = r"[<>≤≥]?\s*\d+(?:[.,]\d+)?"
RANGE = re.compile(r"(" + NUM + r"\s*[-–]\s*" + NUM + r")")
QUAL = re.compile(
    r"^(non[- ]?reactive|not detected|negative|positive|reactive|detected|absent|present|trace|\+{1,4})(?:\b|$)",
    re.I,
)
UNITS = re.compile(
    r"(?<![A-Za-z])((?:[x×]\s*)?10(?:\^?[369]|[³⁶⁹])\s*/\s*(?:u|µ|μ)?[lL]|(?:[munpµμdkM]?(?:mol|g|IU|U|Eq)|cells?)\s*/\s*(?:dL|mL|L|uL|µL|μL|mm3|cumm)|(?:mm\s*/\s*hr|mIU/L|ng/mL|pg/mL|fL|pg|sec(?:onds)?|%|ratio))(?![A-Za-z])",
    re.I,
)
HEADER = {
    "test": r"^(?:test|investigation|parameter|analyte)s?$",
    "value": r"^(?:result|value|observed)s?$",
    "unit": r"^units?$",
    "reference_range": r"^(?:reference|ref\.?|normal|biological)$",
}


def evidence(line, page, method, quote=None):
    return Evidence(
        quote=(quote or line.text)[:1200],
        page=page,
        method=method,
        bbox=line.bbox if any(line.bbox) else None,
    )


def fields(value_text, unit_text=None, reference_text=None):
    value_text = value_text.strip().lstrip(":| ")
    reference = reference_text.strip() if reference_text else None
    # Remove explicit ranges before looking for the measured value, regardless of order.
    ranges = list(RANGE.finditer(value_text))
    if ranges:
        if not reference:
            reference = ranges[0].group(1).strip()
        value_text = RANGE.sub(" ", value_text)
    unit_match = UNITS.search(unit_text or value_text)
    unit = unit_match.group(1).strip() if unit_match else None
    if unit_match and not unit_text:
        value_text = (
            value_text[: unit_match.start()] + " " + value_text[unit_match.end() :]
        )
    value_text = value_text.strip(" :|()[]")
    number = re.match(NUM + r"(?=\s|$|[(*])", value_text)
    qualitative = QUAL.match(value_text)
    value = (
        number.group().strip()
        if number
        else qualitative.group().strip()
        if qualitative
        else None
    )
    return value, unit, reference


def parse_lines(lines, page, method):
    findings, columns = [], None
    for line in lines:
        header = {}
        for index, word in enumerate(line.words):
            if (
                index
                and re.match(r"^(?:range|interval|values?)$", word.text, re.I)
                and re.match(
                    r"^(?:reference|normal|biological)$",
                    line.words[index - 1].text,
                    re.I,
                )
            ):
                continue
            for key, pattern in HEADER.items():
                if re.fullmatch(pattern, word.text, re.I):
                    header.setdefault(key, word.x)
        if (
            "value" in header
            and ("unit" in header or "reference_range" in header)
            and any(w.x for w in line.words)
        ):
            columns = dict(sorted(header.items(), key=lambda item: item[1]))
            continue
        text = re.sub(r"^\s*\d+[.)]\s*", "", line.text).strip()
        name, tail = None, None
        for canonical, pattern in PATTERNS:
            match = pattern.match(text)
            if match:
                name, tail = canonical, text[match.end() :]
                break
        cells = {}
        if columns:
            positions = list(columns.items())
            for word in line.words:
                key = "test"
                for candidate, x in positions:
                    if word.x >= x - 0.018:
                        key = candidate
                cells[key] = (cells.get(key, "") + " " + word.text).strip()
            if (
                not name
                and cells.get("test")
                and cells.get("value")
                and (re.match(NUM, cells["value"]) or QUAL.match(cells["value"]))
            ):
                candidate = cells["test"]
                if 2 < len(candidate) < 70 and not re.search(
                    r"patient|name|date|age|doctor|address|phone|sample id",
                    candidate,
                    re.I,
                ):
                    name = candidate
            if name and cells.get("value"):
                value, unit, reference = fields(
                    cells["value"], cells.get("unit"), cells.get("reference_range")
                )
            elif name:
                value, unit, reference = fields(tail or "")
            else:
                continue
        elif name:
            value, unit, reference = fields(tail or "")
        else:
            continue
        if (
            name in ("Platelets", "Red blood cells", "White blood cells")
            and unit == "%"
        ):
            unit = None
        review = method == "ocr" or not all([value, unit, reference])
        reason = (
            "Check OCR against the original."
            if method == "ocr"
            else "Some fields could not be read; do not fill them by inference."
            if review
            else None
        )
        findings.append(
            Finding(
                test_name=name,
                value=value,
                unit=unit,
                reference_range=reference,
                source_page=page,
                method=method,
                needs_review=review,
                evidence=evidence(line, page, method),
                review_reason=reason,
            )
        )
        if len(findings) >= 200:
            break
    return findings


def parse_text(text, page, method):
    return parse_lines(text_lines(text), page, method)
