"""Small, attributed educational catalog. No prescribing or treatment instructions."""

import json
from pathlib import Path

# Original short paraphrases of the linked public health topic pages.
ROWS = [
    (
        "Fungal infection",
        "Fungal infection",
        "fungalinfections",
        "Fungi can cause infections in the skin and other parts of the body.",
    ),
    (
        "Allergy",
        "Allergy",
        "allergy",
        "An allergy is an immune response to a substance that is usually harmless.",
    ),
    (
        "GERD",
        "Gastroesophageal reflux disease",
        "gerd",
        "GERD involves stomach contents flowing back into the esophagus.",
    ),
    (
        "Chronic cholestasis",
        "Cholestasis",
        "liverdiseases",
        "Cholestasis describes reduced or blocked bile flow. This broad dataset label does not identify its cause.",
    ),
    (
        "Drug Reaction",
        "Drug reaction",
        "drugreactions",
        "Medicines can cause unwanted reactions. A symptom match cannot establish which medicine caused a reaction.",
    ),
    (
        "Peptic ulcer diseae",
        "Peptic ulcer disease",
        "pepticulcer",
        "Peptic ulcers are sores in the lining of the stomach or upper small intestine.",
    ),
    (
        "AIDS",
        "HIV / AIDS",
        "hiv",
        "HIV affects the immune system. AIDS is its most advanced stage; symptoms alone cannot establish HIV status.",
    ),
    (
        "Diabetes",
        "Diabetes",
        "diabetes",
        "Diabetes is associated with persistently high blood glucose. Diagnosis requires appropriate clinical testing.",
    ),
    (
        "Gastroenteritis",
        "Gastroenteritis",
        "gastroenteritis",
        "Gastroenteritis is inflammation of the stomach and intestines, often associated with diarrhea and vomiting.",
    ),
    (
        "Bronchial Asthma",
        "Asthma",
        "asthma",
        "Asthma affects the airways and can cause episodes of breathing difficulty.",
    ),
    (
        "Hypertension",
        "High blood pressure",
        "highbloodpressure",
        "High blood pressure often has no symptoms and is identified by measuring blood pressure.",
    ),
    (
        "Migraine",
        "Migraine",
        "migraine",
        "Migraine can involve recurrent headache and sensitivity to light or sound, sometimes with nausea.",
    ),
    (
        "Cervical spondylosis",
        "Cervical spondylosis",
        "neckinjuriesanddisorders",
        "Cervical spondylosis involves age-related changes in the neck portion of the spine.",
    ),
    (
        "Paralysis (brain hemorrhage)",
        "Hemorrhagic stroke",
        "stroke",
        "Bleeding in the brain can cause a stroke. Sudden weakness or difficulty speaking needs emergency assessment.",
    ),
    (
        "Jaundice",
        "Jaundice",
        "jaundice",
        "Jaundice is yellowing of the skin or eyes due to bilirubin buildup. It is a sign with several possible causes.",
    ),
    (
        "Malaria",
        "Malaria",
        "malaria",
        "Malaria is a parasitic infection usually spread through bites from infected mosquitoes.",
    ),
    (
        "Chicken pox",
        "Chickenpox",
        "chickenpox",
        "Chickenpox is a contagious viral infection that can cause an itchy blistering rash.",
    ),
    (
        "Dengue",
        "Dengue",
        "dengue",
        "Dengue is a viral infection spread by infected mosquitoes.",
    ),
    (
        "Typhoid",
        "Typhoid fever",
        "typhoidfever",
        "Typhoid fever is a bacterial infection that can spread through contaminated food or water.",
    ),
    (
        "hepatitis A",
        "Hepatitis A",
        "hepatitisa",
        "Hepatitis A is a viral liver infection that can spread through contaminated food or water.",
    ),
    (
        "Hepatitis B",
        "Hepatitis B",
        "hepatitisb",
        "Hepatitis B is a viral liver infection that may be short-term or chronic.",
    ),
    (
        "Hepatitis C",
        "Hepatitis C",
        "hepatitisc",
        "Hepatitis C is a bloodborne viral liver infection that can become chronic.",
    ),
    (
        "Hepatitis D",
        "Hepatitis D",
        "hepatitis",
        "Hepatitis D is a viral liver infection that occurs in people with hepatitis B.",
    ),
    (
        "Hepatitis E",
        "Hepatitis E",
        "hepatitis",
        "Hepatitis E is a viral liver infection often spread through contaminated water.",
    ),
    (
        "Alcoholic hepatitis",
        "Alcohol-associated hepatitis",
        "alcoholicfattyliverdisease",
        "Alcohol-associated hepatitis is liver inflammation associated with alcohol use.",
    ),
    (
        "Tuberculosis",
        "Tuberculosis",
        "tuberculosis",
        "Tuberculosis is a bacterial infection that commonly affects the lungs and can spread through the air.",
    ),
    (
        "Common Cold",
        "Common cold",
        "commoncold",
        "The common cold is a viral infection of the upper respiratory tract.",
    ),
    (
        "Pneumonia",
        "Pneumonia",
        "pneumonia",
        "Pneumonia is an infection in the lungs that can have several causes.",
    ),
    (
        "Dimorphic hemmorhoids(piles)",
        "Hemorrhoids",
        "hemorrhoids",
        "Hemorrhoids are swollen veins around the anus or lower rectum. The original dataset uses a nonstandard label.",
    ),
    (
        "Heart attack",
        "Heart attack",
        "heartattack",
        "A heart attack occurs when blood flow to part of the heart is blocked. Suspected symptoms require emergency help.",
    ),
    (
        "Varicose veins",
        "Varicose veins",
        "varicoseveins",
        "Varicose veins are enlarged, twisted veins, most commonly in the legs.",
    ),
    (
        "Hypothyroidism",
        "Underactive thyroid",
        "hypothyroidism",
        "Hypothyroidism means the thyroid does not produce enough thyroid hormone.",
    ),
    (
        "Hyperthyroidism",
        "Overactive thyroid",
        "hyperthyroidism",
        "Hyperthyroidism means the thyroid produces more thyroid hormone than the body needs.",
    ),
    (
        "Hypoglycemia",
        "Low blood glucose",
        "hypoglycemia",
        "Hypoglycemia means blood glucose is too low; a symptom model cannot measure glucose.",
    ),
    (
        "Osteoarthristis",
        "Osteoarthritis",
        "osteoarthritis",
        "Osteoarthritis involves changes in joint tissues, including cartilage, and may cause pain and stiffness.",
    ),
    (
        "Arthritis",
        "Arthritis",
        "arthritis",
        "Arthritis is a broad term for conditions affecting joints. Different forms have different causes.",
    ),
    (
        "(vertigo) Paroymsal Positional Vertigo",
        "Positional vertigo",
        "dizzinessandvertigo",
        "Positional vertigo can produce brief episodes of a spinning sensation when head position changes.",
    ),
    (
        "Acne",
        "Acne",
        "acne",
        "Acne occurs when hair follicles become blocked and can cause several types of skin lesions.",
    ),
    (
        "Urinary tract infection",
        "Urinary tract infection",
        "urinarytractinfections",
        "A urinary tract infection affects part of the urinary system, often the bladder.",
    ),
    (
        "Psoriasis",
        "Psoriasis",
        "psoriasis",
        "Psoriasis is a chronic inflammatory skin condition that can cause patches of thick, scaly skin.",
    ),
    (
        "Impetigo",
        "Impetigo",
        "impetigo",
        "Impetigo is a contagious bacterial skin infection that commonly causes sores and crusting.",
    ),
]
root = Path(__file__).resolve().parents[1]
catalog = [
    {
        "condition": condition,
        "label": label,
        "description": description,
        "sources": [
            {
                "id": f"medlineplus-{slug}",
                "title": f"MedlinePlus: {label}",
                "url": f"https://medlineplus.gov/{slug}.html",
                "publisher": "U.S. National Library of Medicine",
                "accessed": "2026-10-11",
            }
        ],
    }
    for condition, label, slug, description in ROWS
]

for entry in catalog:
    exact = {
        "Chronic cholestasis": "000215",
        "Cervical spondylosis": "000436",
        "(vertigo) Paroymsal Positional Vertigo": "001420",
    }
    if entry["condition"] in exact:
        entry["sources"][0]["url"] = (
            f"https://medlineplus.gov/ency/article/{exact[entry['condition']]}.htm"
        )
    if entry["condition"] == "Typhoid":
        entry["sources"][0]["url"] = "https://medlineplus.gov/ency/article/001332.htm"
    if entry["condition"] == "Alcoholic hepatitis":
        entry["sources"][0]["url"] = "https://medlineplus.gov/ency/article/000281.htm"
(root / "shared/catalog.json").write_text(
    json.dumps(catalog, indent=2, ensure_ascii=False) + "\n"
)
