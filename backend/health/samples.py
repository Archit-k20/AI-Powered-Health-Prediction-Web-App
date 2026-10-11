import json
import os
from .model import ROOT, CATALOG
from .schemas import SampleExplanation, ExplanationPayload
import httpx

SAMPLES = {
    s["id"]: s for s in json.loads((ROOT / "shared/sample-inputs.json").read_text())
}


def prepared_explanation(sample_id):
    """Only accepts an allowlisted ID. No request text, reports or health session enters here."""
    sample = SAMPLES[sample_id]
    passage = CATALOG[sample["condition"]]
    allowed = [s["id"] for s in passage["sources"]]
    return SampleExplanation(
        sample_id=sample_id,
        mode="prepared",
        text=f"In this fictional walkthrough, the selected symptoms illustrate how a small dataset can associate a pattern with {passage['label'].lower()}. {passage['description']} Other explanations are possible. The classifier's ranking is not a diagnosis and the source describes the condition; it does not validate the model.",
        source_ids=allowed,
        notice="Prepared educational explanation for a fictional case. No visitor input was used.",
    )


async def explain_sample(sample_id):
    sample = SAMPLES[sample_id]
    passage = CATALOG[sample["condition"]]
    allowed = [s["id"] for s in passage["sources"]]
    prepared = prepared_explanation(sample_id)
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return prepared
    prompt = {
        "task": "Explain this predefined fictional example in under 120 words. No diagnosis, treatment or personal advice. Use only supplied passages. Return JSON with text and source_ids; cite only supplied IDs. State that this is fictional and model matches are not diagnoses.",
        "fictional_case": sample,
        "retrieved_catalog_passage": passage,
    }
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": key},
                json={
                    "contents": [{"parts": [{"text": json.dumps(prompt)}]}],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "temperature": 0.2,
                        "maxOutputTokens": 600,
                    },
                },
            )
            response.raise_for_status()
            payload = ExplanationPayload.model_validate_json(
                response.json()["candidates"][0]["content"]["parts"][0]["text"]
            )
            if set(payload.source_ids) - set(allowed):
                return prepared
            return SampleExplanation(
                sample_id=sample_id,
                mode="gemini",
                text=payload.text,
                source_ids=payload.source_ids,
                notice="AI-generated explanation of a predefined fictional case only. Verify against the linked sources.",
            )
    except Exception:
        # Do not log prompts, responses, credentials or provider errors.
        return prepared
