"""Conservative, source-linked mention extraction; not diagnosis or causal inference."""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from .lab_parser import evidence
from .schemas import ReportMention

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = json.loads((ROOT / "shared/symptoms.json").read_text())["symptoms"]
CATALOG = json.loads((ROOT / "shared/catalog.json").read_text())
# These registry features are diagnoses/findings, not patient-reported symptoms.
NON_SYMPTOMS = {
    "acute_liver_failure",
    "fluid_overload",
    "toxic_look_(typhos)",
    "coma",
    "stomach_bleeding",
    "history_of_alcohol_consumption",
}
AMBIGUOUS = {"hot", "temperature", "fever"}
PATTERNS = []
for s in REGISTRY:
    aliases = {s["id"].replace("_", " "), s["label"], *s["aliases"]} - AMBIGUOUS
    for alias in sorted(aliases, key=len, reverse=True):
        PATTERNS.append(
            (
                re.compile(
                    r"(?<!\w)" + re.escape(alias).replace(r"\ ", r"[\s-]+") + r"(?!\w)",
                    re.I,
                ),
                s,
            )
        )
PATTERNS.sort(key=lambda pair: -len(pair[0].pattern))

NEGATED = re.compile(
    r"\b(?:no(?: evidence of)?|not having|(?:does not|doesn['’]t|did not|didn['’]t) (?:have|report|experience)|never had|denies?|denied|without|negative for|free of|absence of)\b",
    re.I,
)
UNCERTAIN = re.compile(
    r"\b(?:possible|possibly|suspected|suspect|may|might|cannot (?:be )?exclude[ds]?|cannot rule out|rule out|r/o|\?)\b",
    re.I,
)
FAMILY = re.compile(
    r"\b(?:family|mother|father|sister|brother|parent|grandmother|grandfather)\b", re.I
)
HISTORY = re.compile(
    r"\b(?:past history|previous(?:ly)?|histor(?:y|ical)|last year|years? ago|as a child|prior episode)\b",
    re.I,
)
RESOLVED = re.compile(r"\b(?:resolved|subsided|no longer|recovered from)\b", re.I)
HYPOTHETICAL = re.compile(
    r"\b(?:if|should .{0,30}develop|watch for|monitor for|return (?:if|for)|seek .{0,30}if|risk of|example|e\.g\.|such as|include|can cause|can develop)\b",
    re.I,
)
META = re.compile(
    r"^(?:patient(?: name)?|name|address|date of birth|dob|doctor|consultant|signature|phone|email|report date|collection date|sample id)\s*:",
    re.I,
)
HEADINGS = {
    "symptom": r"^(?:presenting complaint|chief complaints?|symptoms?|history of present illness|hpi|clinical history|clinical indication|complaints?)\s*:?\s*",
    "history": r"^(?:past medical history|past history|medical history|family history)\s*:?\s*",
    "diagnosis": r"^(?:diagnos(?:is|es)|assessment|final diagnosis|provisional diagnosis|discharge diagnosis)\s*:?\s*",
    "finding": r"^(?:findings?|impression|conclusion)\s*:?\s*",
    "instruction": r"^(?:instructions?|advice|plan|recommendations?|medications?|treatment|prescription)\s*:?\s*",
}


def classify_document(text, has_labs=False):
    lower = text.lower()
    markers = [
        "report prepared for",
        "report typed",
        "who requested",
        "record information",
        "sources of information",
        "full name",
        "who provided consent",
        "presenting complaint",
    ]
    # Fuzzy matching applies only to template headings, never to clinical inputs.
    normalized_lines = [
        re.sub(r"[^a-z ]", "", line).strip() for line in lower.splitlines()
    ]
    template = (
        sum(
            x in lower
            or any(
                SequenceMatcher(None, x, line[: len(x) + 3]).ratio() >= 0.83
                for line in normalized_lines
            )
            for x in markers
        )
        >= 3
    )
    clinical_claim = re.search(
        r"\b(?:patient (?:reports|has|denies)|complains of|diagnosed with|result\s*[:=]\s*\d)",
        lower,
    )
    if template and not clinical_claim and not has_labs:
        return "template"
    if re.search(r"discharge (?:summary|diagnosis|note)", lower):
        return "discharge_summary"
    if re.search(
        r"\b(?:radiology|x[- ]ray|mri|ct scan|ultrasound|sonography|computed tomography)\b",
        lower,
    ) and re.search(r"\b(?:findings?|impression|report)\b", lower):
        return "radiology"
    if has_labs:
        return "laboratory"
    if re.search(
        r"\b(?:patient|symptoms?|complaints?|clinical|medical report|diagnosis|assessment|consultation)\b",
        lower,
    ):
        return "clinical_note"
    return "unknown"


def contextualize(text, start, end, section, section_text):
    # A contrast/new explicit patient assertion ends preceding negation or family scope.
    prefix = text[:start]
    context_prefix = re.split(r"\b(?:but|however|although)\b", prefix, flags=re.I)[-1]
    if re.search(r"\bpatient\b", prefix, re.I):
        context_prefix = re.split(r"\bpatient\b", context_prefix, flags=re.I)[-1]
    scope = re.split(r"\b(?:reports|complains of|has)\b", context_prefix, flags=re.I)[
        -1
    ]
    following = text[end : end + 45]
    context = context_prefix + " " + following
    polarity = (
        "uncertain"
        if UNCERTAIN.search(context)
        else "negated"
        if NEGATED.search(scope)
        or re.match(r"\s*(?:is |was |: )?(?:absent|denied|negative)\b", following, re.I)
        else "affirmed"
    )
    if re.search(r"\b(?:not only|no change in)\b", scope, re.I):
        polarity = "uncertain"
    subject = (
        "family"
        if FAMILY.search(context_prefix) or "family history" in section_text.lower()
        else "patient"
    )
    timing = (
        "resolved"
        if RESOLVED.search(context)
        else "hypothetical"
        if HYPOTHETICAL.search(context_prefix)
        or section == "instruction"
        or "?" in text
        else "historical"
        if HISTORY.search(context_prefix) or section == "history"
        else "unspecified"
    )
    return polarity, timing, subject


def extract_mentions(lines, page, method, document_type):
    if document_type == "template":
        return []
    mentions, section, section_text = [], "", ""

    def add(label, category, symptom_id, line, sentence, start=0, end=None):
        polarity, timing, subject = contextualize(
            sentence, start, end or len(sentence), section, section_text
        )
        eligible = (
            category == "symptom"
            and symptom_id is not None
            and polarity == "affirmed"
            and timing == "unspecified"
            and subject == "patient"
        )
        reason = (
            "Check the wording and confirm this symptom is happening to you now."
            if eligible
            else "Kept as document context; not added to the symptom model."
        )
        if category == "symptom" and symptom_id is None:
            reason = "No unambiguous supported symptom mapping. Choose the appropriate symptom manually if it applies now."
        mentions.append(
            ReportMention(
                id=f"p{page}-m{len(mentions) + 1}",
                label=label,
                category=category,
                symptom_id=symptom_id,
                polarity=polarity,
                timing=timing,
                subject=subject,
                eligible=eligible,
                review_reason=reason,
                evidence=evidence(line, page, method, sentence),
            )
        )

    for line in lines:
        if META.match(line.text):
            continue
        text = line.text.strip()
        heading = None
        for category, pattern in HEADINGS.items():
            m = re.match(pattern, text, re.I)
            if m:
                heading = category
                section, section_text = category, m.group()
                text = text[m.end() :]
                break
        if not text or len(text) < 3:
            continue
        if heading in ("diagnosis", "history", "finding") and text:
            add(text[:300], heading, None, line, line.text)
        # Paragraphs in these sections stay document claims, never become symptoms.
        if section in ("diagnosis", "finding"):
            if heading is None:
                add(text[:300], section, None, line, text)
            continue
        for sentence in re.split(r"(?<=[.!?;])\s+", text):
            occupied = []
            for pattern, symptom in PATTERNS:
                for match in pattern.finditer(sentence):
                    if any(match.start() < b and match.end() > a for a, b in occupied):
                        continue
                    occupied.append((match.start(), match.end()))
                    category = (
                        "diagnosis" if symptom["id"] in NON_SYMPTOMS else "symptom"
                    )
                    add(
                        symptom["label"],
                        category,
                        symptom["id"] if category == "symptom" else None,
                        line,
                        sentence,
                        match.start(),
                        match.end(),
                    )
            fever = re.search(r"\bfever\b", sentence, re.I)
            if fever and not any(
                fever.start() < b and fever.end() > a for a, b in occupied
            ):
                add(
                    "Fever (severity not specified)",
                    "symptom",
                    None,
                    line,
                    sentence,
                    fever.start(),
                    fever.end(),
                )
            if (
                section == "symptom"
                and not occupied
                and not fever
                and len(sentence) > 3
            ):
                add(sentence[:300], "symptom", None, line, sentence)
            if re.search(r"\b(?:due to|caused by|secondary to)\b", sentence, re.I):
                add(sentence[:300], "cause", None, line, sentence)
            for condition in CATALOG:
                match = re.search(
                    r"\b" + re.escape(condition["label"]) + r"\b", sentence, re.I
                )
                if match and not any(
                    match.start() < b and match.end() > a for a, b in occupied
                ):
                    add(
                        condition["label"],
                        "diagnosis",
                        None,
                        line,
                        sentence,
                        match.start(),
                        match.end(),
                    )
        if len(mentions) >= 100:
            return mentions[:100]
    return mentions
