# Model evaluation

Version: `health-rf-1.0.0`. Dataset SHA-256: `ebbd391c4ba4d64f57a00eb3d0a55f0ca9b920b0c5de6b9af4234e72519c9618`.

4920 raw rows → 304 unique patterns → 243 train / 61 held out; stratified split, seed 42. Four-fold stratified cross-validation uses the training partition only. All 17 symptom columns are included. Disease is exclusively the target. No final refit on the holdout.

| Model | Holdout accuracy | Macro F1 | Top-three recall | CV accuracy |
|---|---:|---:|---:|---:|
| random_forest | 1.0000 | 1.0000 | 1.0000 | 0.9959 |
| logistic_regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Limits

- Educational model; not clinically validated and not a diagnosis.
- The original dataset's clinical provenance, collection process and population are undocumented.
- 4,920 rows collapse to 304 distinct symptom patterns across 41 labels. Templates are highly repetitive.
- Small internal holdout; results do not establish performance on real people or unseen conditions.
- Only recorded symptom presence is modeled. Missing selections are encoded as absent; age, duration and lab values are not model inputs.
- Ranked matches are model similarities, not personal disease probabilities. Some dataset labels describe symptoms or syndromes rather than diagnoses.

## Random forest per-class holdout results

| Dataset label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| (vertigo) Paroymsal Positional Vertigo | 1.000 | 1.000 | 1.000 | 1 |
| AIDS | 1.000 | 1.000 | 1.000 | 1 |
| Acne | 1.000 | 1.000 | 1.000 | 1 |
| Alcoholic hepatitis | 1.000 | 1.000 | 1.000 | 2 |
| Allergy | 1.000 | 1.000 | 1.000 | 1 |
| Arthritis | 1.000 | 1.000 | 1.000 | 1 |
| Bronchial Asthma | 1.000 | 1.000 | 1.000 | 1 |
| Cervical spondylosis | 1.000 | 1.000 | 1.000 | 1 |
| Chicken pox | 1.000 | 1.000 | 1.000 | 2 |
| Chronic cholestasis | 1.000 | 1.000 | 1.000 | 2 |
| Common Cold | 1.000 | 1.000 | 1.000 | 2 |
| Dengue | 1.000 | 1.000 | 1.000 | 2 |
| Diabetes | 1.000 | 1.000 | 1.000 | 2 |
| Dimorphic hemmorhoids(piles) | 1.000 | 1.000 | 1.000 | 1 |
| Drug Reaction | 1.000 | 1.000 | 1.000 | 1 |
| Fungal infection | 1.000 | 1.000 | 1.000 | 1 |
| GERD | 1.000 | 1.000 | 1.000 | 1 |
| Gastroenteritis | 1.000 | 1.000 | 1.000 | 1 |
| Heart attack | 1.000 | 1.000 | 1.000 | 1 |
| Hepatitis B | 1.000 | 1.000 | 1.000 | 2 |
| Hepatitis C | 1.000 | 1.000 | 1.000 | 1 |
| Hepatitis D | 1.000 | 1.000 | 1.000 | 2 |
| Hepatitis E | 1.000 | 1.000 | 1.000 | 2 |
| Hypertension | 1.000 | 1.000 | 1.000 | 1 |
| Hyperthyroidism | 1.000 | 1.000 | 1.000 | 2 |
| Hypoglycemia | 1.000 | 1.000 | 1.000 | 2 |
| Hypothyroidism | 1.000 | 1.000 | 1.000 | 2 |
| Impetigo | 1.000 | 1.000 | 1.000 | 1 |
| Jaundice | 1.000 | 1.000 | 1.000 | 2 |
| Malaria | 1.000 | 1.000 | 1.000 | 2 |
| Migraine | 1.000 | 1.000 | 1.000 | 2 |
| Osteoarthristis | 1.000 | 1.000 | 1.000 | 1 |
| Paralysis (brain hemorrhage) | 1.000 | 1.000 | 1.000 | 1 |
| Peptic ulcer diseae | 1.000 | 1.000 | 1.000 | 1 |
| Pneumonia | 1.000 | 1.000 | 1.000 | 2 |
| Psoriasis | 1.000 | 1.000 | 1.000 | 1 |
| Tuberculosis | 1.000 | 1.000 | 1.000 | 2 |
| Typhoid | 1.000 | 1.000 | 1.000 | 2 |
| Urinary tract infection | 1.000 | 1.000 | 1.000 | 1 |
| Varicose veins | 1.000 | 1.000 | 1.000 | 2 |
| hepatitis A | 1.000 | 1.000 | 1.000 | 2 |

Reproduce with `.venv/bin/python training/train.py`. Full CV folds, per-class baseline and exact dependency versions are in `backend/models/metadata.json`. Artifact checksums are verified before loading. No accuracy claim from this internal dataset should be presented as clinical performance.
