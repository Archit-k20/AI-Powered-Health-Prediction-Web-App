from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Source(StrictModel):
    id: str
    title: str
    url: str
    publisher: str
    accessed: str


class Question(StrictModel):
    id: str
    prompt: str
    options: list[str]
    suggests: str | None


class Symptom(StrictModel):
    id: str
    label: str
    aliases: list[str]


class Registry(StrictModel):
    version: str
    symptoms: list[Symptom]
    questions: list[Question]


class Followup(StrictModel):
    question_id: str = Field(max_length=50)
    answer: str = Field(max_length=50)
    confirmed: bool = False


class ConfirmedReportSymptom(StrictModel):
    symptom_id: str = Field(max_length=80)
    confirmed_current: bool


class AssessmentInput(StrictModel):
    symptoms: list[str] = Field(default_factory=list, max_length=50)
    followups: list[Followup] = Field(default_factory=list, max_length=10)
    report_symptoms: list[ConfirmedReportSymptom] = Field(
        default_factory=list, max_length=50
    )


class Contribution(StrictModel):
    symptom_id: str
    label: str
    present: bool
    contribution: float


class Match(StrictModel):
    rank: int
    condition: str
    label: str
    description: str
    supporting_symptoms: list[str]
    contributions: list[Contribution]
    base_value: float
    model_output: float
    sources: list[Source]


class Assessment(StrictModel):
    model_version: str
    symptoms: list[str]
    context: list[str]
    matches: list[Match]
    limitations: list[str]
    notice: str


class Evidence(StrictModel):
    quote: str
    page: int
    method: Literal["embedded_text", "ocr"]
    # Normalized top-left coordinates: x, y, width, height.
    bbox: list[float] | None = None


class ReportMention(StrictModel):
    id: str
    label: str
    category: Literal["symptom", "diagnosis", "history", "finding", "cause"]
    symptom_id: str | None = None
    polarity: Literal["affirmed", "negated", "uncertain"]
    timing: Literal["unspecified", "historical", "hypothetical", "resolved"]
    subject: Literal["patient", "family", "unknown"]
    eligible: bool = False
    review_reason: str
    evidence: Evidence


class ReportPage(StrictModel):
    page: int
    document_type: Literal[
        "laboratory",
        "clinical_note",
        "discharge_summary",
        "radiology",
        "template",
        "unknown",
    ]
    method: Literal["embedded_text", "ocr"]
    quality: Literal["readable", "review", "unreadable"]
    text: str
    warnings: list[str] = Field(default_factory=list)


class Finding(StrictModel):
    test_name: str
    value: str | None
    unit: str | None
    reference_range: str | None
    source_page: int
    method: Literal["embedded_text", "ocr", "fictional", "manual"]
    needs_review: bool
    evidence: Evidence | None = None
    review_reason: str | None = None


class Extraction(StrictModel):
    findings: list[Finding]
    warnings: list[str]
    pages: int
    partial: bool
    page_details: list[ReportPage] = Field(default_factory=list)
    mentions: list[ReportMention] = Field(default_factory=list)
    pipeline_version: str = "reports-2.0.0"
    notice: str = "Check the original report. Only symptoms you confirm as current can enter the model; lab values and documented diagnoses stay separate. Extraction can miss information."


class Health(StrictModel):
    status: Literal["ready", "degraded"]
    model_version: str | None
    capabilities: dict[str, bool]


class SampleExplanation(StrictModel):
    sample_id: str
    mode: Literal["prepared", "gemini"]
    text: str
    source_ids: list[str]
    notice: str


class ExplanationPayload(StrictModel):
    text: str = Field(min_length=20, max_length=1800)
    source_ids: list[str] = Field(min_length=1, max_length=5)


class ErrorDetail(StrictModel):
    code: str
    message: str


class ErrorResponse(StrictModel):
    error: ErrorDetail


class Metric(StrictModel):
    accuracy: float
    macro_f1: float
    top_three_recall: float
    cross_validation: dict
    per_class: dict


class ModelInfo(StrictModel):
    version: str
    registry_version: str
    registry_sha256: str
    dataset_sha256: str
    seed: int
    raw_rows: int
    unique_patterns: int
    train_count: int
    test_count: int
    evaluation: dict[str, Metric]
    limitations: list[str]
