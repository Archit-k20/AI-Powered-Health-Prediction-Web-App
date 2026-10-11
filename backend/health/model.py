import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

import joblib
import numpy as np
import shap

from .schemas import Assessment, Match, Contribution

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = json.loads((ROOT / "shared/symptoms.json").read_text())
CATALOG = {
    c["condition"]: c for c in json.loads((ROOT / "shared/catalog.json").read_text())
}
SYMPTOMS = {s["id"]: s for s in REGISTRY["symptoms"]}


def normalize(text):
    return re.sub(r"[\s_]+", "_", text.strip().lower()).strip("_")


ALIASES = {}
for symptom in REGISTRY["symptoms"]:
    for alias in [symptom["id"], symptom["label"], *symptom["aliases"]]:
        key = normalize(alias)
        if key in ALIASES and ALIASES[key] != symptom["id"]:
            raise ValueError("Ambiguous symptom alias in registry")
        ALIASES[key] = symptom["id"]


class InputError(ValueError):
    pass


class ModelService:
    def __init__(self, directory=None):
        directory = Path(directory or ROOT / "backend/models")
        self.metadata = json.loads((directory / "metadata.json").read_text())
        m = self.metadata
        if (
            hashlib.sha256((ROOT / "shared/symptoms.json").read_bytes()).hexdigest()
            != m["registry_sha256"]
        ):
            raise ValueError("Registry does not match model")
        if (
            hashlib.sha256((directory / "model.joblib").read_bytes()).hexdigest()
            != m["artifact_sha256"]
        ):
            raise ValueError("Model checksum mismatch")
        for package, version in m["dependencies"].items():
            if importlib.metadata.version(package) != version:
                raise ValueError("Model dependency version mismatch")
        self.model = joblib.load(directory / "model.joblib")
        self.features = m["features"]
        if (
            self.features != list(SYMPTOMS)
            or self.model.n_features_in_ != len(self.features)
            or self.model.classes_.tolist() != m["classes"]
        ):
            raise ValueError("Model schema mismatch")
        if set(m["classes"]) - set(CATALOG):
            raise ValueError("Condition missing from catalog")
        self.explainer = shap.TreeExplainer(self.model)

    def assess(self, payload):
        selected = []
        if any(not item.confirmed_current for item in payload.report_symptoms):
            raise InputError(
                "Confirm that each report symptom is happening to you now before including it."
            )
        for value in [
            *payload.symptoms,
            *(item.symptom_id for item in payload.report_symptoms),
        ]:
            key = ALIASES.get(normalize(value))
            if not key:
                raise InputError(
                    "One or more symptoms are unsupported. Choose symptoms from the search list."
                )
            if key not in selected:
                selected.append(key)
        context = []
        if payload.report_symptoms:
            context.append(
                "Report-derived symptoms were explicitly confirmed as current by the visitor. Lab values and documented diagnoses were not used."
            )
        questions = {q["id"]: q for q in REGISTRY["questions"]}
        seen = set()
        for answer in payload.followups:
            q = questions.get(answer.question_id)
            if not q or answer.answer not in q["options"] or answer.question_id in seen:
                raise InputError("Review the follow-up answers and try again.")
            seen.add(answer.question_id)
            context.append(f"{q['prompt']} {answer.answer}")
            if (
                q["suggests"]
                and answer.answer == "Yes"
                and answer.confirmed
                and q["suggests"] not in selected
            ):
                selected.append(q["suggests"])
        if len(selected) > 50:
            raise InputError("Use at most 50 distinct symptoms in an assessment.")
        if len(selected) < 2:
            raise InputError(
                "Select at least two distinct supported symptoms. A single symptom is not enough for this demonstration."
            )
        x = np.array([[int(f in selected) for f in self.features]], dtype=np.uint8)
        outputs = self.model.predict_proba(x)[0]
        values = self.explainer.shap_values(x)
        matches = []
        for rank, class_index in enumerate(
            np.argsort(-outputs, kind="stable")[:3], start=1
        ):
            # SHAP >= 0.45 returns [sample, feature, class]; older releases returned one array per class.
            contributions = (
                values[class_index][0]
                if isinstance(values, list)
                else values[0, :, class_index]
            )
            assert len(contributions) == len(self.features)
            order = np.argsort(-np.abs(contributions), kind="stable")
            catalog = CATALOG[str(self.model.classes_[class_index])]
            matches.append(
                Match(
                    rank=rank,
                    condition=catalog["condition"],
                    label=catalog["label"],
                    description=catalog["description"],
                    supporting_symptoms=[
                        f
                        for i, f in enumerate(self.features)
                        if f in selected and contributions[i] > 0
                    ],
                    contributions=[
                        Contribution(
                            symptom_id=self.features[i],
                            label=SYMPTOMS[self.features[i]]["label"],
                            present=self.features[i] in selected,
                            contribution=float(contributions[i]),
                        )
                        for i in order[:8]
                    ],
                    base_value=float(self.explainer.expected_value[class_index]),
                    model_output=float(outputs[class_index]),
                    sources=catalog["sources"],
                )
            )
        return Assessment(
            model_version=self.metadata["version"],
            symptoms=selected,
            context=context,
            matches=matches,
            limitations=self.metadata["limitations"],
            notice="Educational exploration only. These are ranked dataset matches, not a diagnosis or personal disease risk. For severe or sudden symptoms, seek urgent medical help; do not rely on this demo.",
        )
