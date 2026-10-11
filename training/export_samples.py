import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.health.model import ModelService
from backend.health.samples import SAMPLES, prepared_explanation
from backend.health.schemas import AssessmentInput

service = ModelService()
results = []
for sample in SAMPLES.values():
    assessment = service.assess(
        AssessmentInput(
            symptoms=sample["symptoms"],
            followups=[
                {
                    "question_id": "duration",
                    "answer": sample["duration"],
                    "confirmed": False,
                }
            ],
        )
    )
    assert assessment.matches[0].condition == sample["condition"], (
        "Fictional source must match the actual model result"
    )
    explanation = prepared_explanation(sample["id"])
    results.append(
        {
            **sample,
            "assessment": assessment.model_dump(),
            "explanation": explanation.model_dump(),
        }
    )
(ROOT / "shared/samples.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False) + "\n"
)
from backend.app import app

(ROOT / "shared/openapi.json").write_text(json.dumps(app.openapi(), indent=2) + "\n")
print("Exported three reproducible sample assessments and OpenAPI.")
