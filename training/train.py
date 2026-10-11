"""Reproducible evaluation: deduplicate before splitting; never fit on holdout."""

import csv
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

ROOT = Path(__file__).resolve().parents[1]
VERSION = "health-rf-1.0.0"


def normalize(value):
    return re.sub(r"[\s_]+", "_", value.strip().lower()).strip("_")


LABELS = {
    "spotting_urination": "Spotting when urinating",
    "dischromic_patches": "Discoloured skin patches",
    "extra_marital_contacts": "Sexual exposure concern",
    "foul_smell_of_urine": "Strong-smelling urine",
    "toxic_look_(typhos)": "Very unwell appearance",
    "scurring": "Scaly skin",
    "high_fever": "High fever",
    "breathlessness": "Shortness of breath",
    "fluid_overload": "Fluid retention",
    "fluid_overload.1": "Fluid retention (dataset variant)",
    "yellow_crust_ooze": "Yellow crust or ooze",
    "continuous_sneezing": "Frequent sneezing",
    "pain_during_bowel_movements": "Pain during bowel movements",
}
ALIASES = {
    "high_fever": ["fever", "temperature", "hot"],
    "breathlessness": [
        "shortness of breath",
        "difficulty breathing",
        "breathing difficulty",
        "dyspnea",
    ],
    "headache": ["head pain"],
    "skin_rash": ["rash", "skin eruption"],
    "fatigue": ["tired", "tiredness", "exhaustion"],
    "diarrhoea": ["diarrhea", "loose stools"],
    "stomach_pain": ["stomach ache", "stomachache"],
    "abdominal_pain": ["belly pain", "abdominal cramps"],
    "vomiting": ["throwing up", "being sick"],
    "burning_micturition": ["painful urination", "burning urination", "dysuria"],
    "continuous_sneezing": ["sneezing"],
    "runny_nose": ["running nose", "rhinorrhea"],
    "chest_pain": ["chest discomfort"],
    "muscle_pain": ["body aches", "myalgia"],
    "dizziness": ["lightheadedness", "dizzy"],
    "yellowish_skin": ["yellow skin"],
    "loss_of_appetite": ["no appetite", "poor appetite"],
    "joint_pain": ["aching joints"],
    "itching": ["itchy", "pruritus"],
    "nausea": ["feeling sick"],
    "cough": ["coughing"],
    "mild_fever": ["low grade fever"],
    "swelled_lymph_nodes": ["swollen glands", "swollen lymph nodes"],
    "constipation": ["difficulty passing stool"],
    "weight_loss": ["losing weight"],
    "watering_from_eyes": ["watery eyes"],
}
LABELS.update(
    {
        "burning_micturition": "Burning when urinating",
        "altered_sensorium": "Confusion or reduced awareness",
        "cold_hands_and_feets": "Cold hands and feet",
        "continuous_feel_of_urine": "Persistent urge to urinate",
        "distention_of_abdomen": "Abdominal swelling",
        "lethargy": "Low energy",
        "malaise": "Feeling generally unwell",
        "mucoid_sputum": "Mucus in coughed-up phlegm",
        "muscle_wasting": "Loss of muscle",
        "nodal_skin_eruptions": "Small raised skin lesions",
        "passage_of_gases": "Passing gas",
        "polyuria": "Passing urine frequently",
        "rusty_sputum": "Rust-coloured phlegm",
        "silver_like_dusting": "Silvery skin scales",
        "swelled_lymph_nodes": "Swollen lymph nodes",
        "swollen_extremeties": "Swollen hands or feet",
        "swelling_joints": "Swollen joints",
        "spinning_movements": "Spinning sensation",
        "pus_filled_pimples": "Pus-filled pimples",
        "belly_pain": "Belly pain",
        "puffy_face_and_eyes": "Puffy face and eyes",
    }
)
ALIASES["abdominal_pain"] = ["abdominal cramps"]
QUESTIONS = [
    {
        "id": "duration",
        "prompt": "How long have you noticed these symptoms?",
        "options": [
            "Less than a day",
            "1–3 days",
            "4–7 days",
            "More than a week",
            "Not sure",
        ],
        "suggests": None,
    },
    {
        "id": "fever_check",
        "prompt": "Have you also noticed a high fever?",
        "options": ["Yes", "No", "Not sure"],
        "suggests": "high_fever",
    },
    {
        "id": "breathing_check",
        "prompt": "Have you also noticed shortness of breath?",
        "options": ["Yes", "No", "Not sure"],
        "suggests": "breathlessness",
    },
]


def load_patterns():
    with (ROOT / "data/dataset.csv").open() as f:
        reader = csv.DictReader(f)
        columns = [f"Symptom_{i}" for i in range(1, 18)]
        assert all(c in reader.fieldnames for c in columns)
        rows = list(reader)
    patterns = {}
    for row in rows:
        key = tuple(sorted({normalize(row[c]) for c in columns if row[c].strip()}))
        label = " ".join(row["Disease"].split())
        assert key not in patterns or patterns[key] == label, "Conflicting labels"
        patterns[key] = label
    features = sorted({s for key in patterns for s in key})
    items = sorted(patterns.items())
    X = np.array(
        [[int(f in key) for f in features] for key, _ in items], dtype=np.uint8
    )
    y = np.array([label for _, label in items])
    hashes = [hashlib.sha256("|".join(key).encode()).hexdigest() for key, _ in items]
    return X, y, features, hashes, len(rows)


def train():
    X, y, features, hashes, raw_count = load_patterns()
    train_ids, test_ids = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    registry = {
        "version": "symptoms-1.0.0",
        "symptoms": [
            {
                "id": f,
                "label": LABELS.get(f, f.replace("_", " ").capitalize()),
                "aliases": list(
                    dict.fromkeys([f.replace("_", " "), *ALIASES.get(f, [])])
                ),
            }
            for f in features
        ],
        "questions": QUESTIONS,
    }
    registry_bytes = (
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n"
    ).encode()
    (ROOT / "shared/symptoms.json").write_bytes(registry_bytes)
    limitations = [
        "Educational model; not clinically validated and not a diagnosis.",
        "The original dataset's clinical provenance, collection process and population are undocumented.",
        f"{raw_count:,} rows collapse to {len(y)} distinct symptom patterns across {len(set(y))} labels. Templates are highly repetitive.",
        "Small internal holdout; results do not establish performance on real people or unseen conditions.",
        "Only recorded symptom presence is modeled. Missing selections are encoded as absent; age, duration and lab values are not model inputs.",
        "Ranked matches are model similarities, not personal disease probabilities. Some dataset labels describe symptoms or syndromes rather than diagnoses.",
    ]
    results = {}
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=1
        ),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=42),
    }
    for name, model in models.items():
        cv = cross_validate(
            model,
            X[train_ids],
            y[train_ids],
            cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=42),
            scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
            n_jobs=1,
        )
        model.fit(X[train_ids], y[train_ids])
        predicted = model.predict(X[test_ids])
        results[name] = {
            "accuracy": accuracy_score(y[test_ids], predicted),
            "macro_f1": f1_score(y[test_ids], predicted, average="macro"),
            "top_three_recall": top_k_accuracy_score(
                y[test_ids],
                model.predict_proba(X[test_ids]),
                k=3,
                labels=model.classes_,
            ),
            "cross_validation": {
                k.replace("test_", ""): {
                    "mean": float(cv[k].mean()),
                    "folds": cv[k].tolist(),
                }
                for k in ["test_accuracy", "test_macro_f1"]
            },
            "per_class": classification_report(
                y[test_ids], predicted, output_dict=True, zero_division=0
            ),
        }
    out = ROOT / "backend/models"
    out.mkdir(exist_ok=True)
    joblib.dump(models["random_forest"], out / "model.joblib", compress=3)
    metadata = {
        "version": VERSION,
        "registry_version": registry["version"],
        "registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "dataset_sha256": hashlib.sha256(
            (ROOT / "data/dataset.csv").read_bytes()
        ).hexdigest(),
        "artifact_sha256": hashlib.sha256(
            (out / "model.joblib").read_bytes()
        ).hexdigest(),
        "features": features,
        "classes": models["random_forest"].classes_.tolist(),
        "dependencies": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "scikit-learn", "shap", "joblib"]
        },
        "seed": 42,
        "raw_rows": raw_count,
        "unique_patterns": len(y),
        "train_count": len(train_ids),
        "test_count": len(test_ids),
        "train_pattern_hashes": [hashes[i] for i in train_ids],
        "test_pattern_hashes": [hashes[i] for i in test_ids],
        "evaluation": results,
        "limitations": limitations,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    public = {
        k: v
        for k, v in metadata.items()
        if k
        not in [
            "train_pattern_hashes",
            "test_pattern_hashes",
            "features",
            "classes",
            "dependencies",
            "artifact_sha256",
        ]
    }
    (ROOT / "shared/model-info.json").write_text(json.dumps(public, indent=2) + "\n")
    lines = [
        "# Model evaluation",
        "",
        f"Version: `{VERSION}`. Dataset SHA-256: `{metadata['dataset_sha256']}`.",
        "",
        f"{raw_count} raw rows → {len(y)} unique patterns → {len(train_ids)} train / {len(test_ids)} held out; stratified split, seed 42. Four-fold stratified cross-validation uses the training partition only. All 17 symptom columns are included. Disease is exclusively the target. No final refit on the holdout.",
        "",
        "| Model | Holdout accuracy | Macro F1 | Top-three recall | CV accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['top_three_recall']:.4f} | {r['cross_validation']['accuracy']['mean']:.4f} |"
        )
    lines += [
        "",
        "## Limits",
        "",
        *[f"- {x}" for x in limitations],
        "",
        "## Random forest per-class holdout results",
        "",
        "| Dataset label | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in results["random_forest"]["per_class"].items():
        if name in metadata["classes"]:
            lines.append(
                f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {r['support']:.0f} |"
            )
    lines += [
        "",
        "Reproduce with `.venv/bin/python training/train.py`. Full CV folds, per-class baseline and exact dependency versions are in `backend/models/metadata.json`. Artifact checksums are verified before loading. No accuracy claim from this internal dataset should be presented as clinical performance.",
    ]
    (ROOT / "docs/EVALUATION.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                k: {
                    m: v
                    for m, v in r.items()
                    if m not in ["per_class", "cross_validation"]
                }
                for k, r in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    train()
