"""
Generates hard-negative training examples: notes with specific clinical
detail that's actually irrelevant to the billed code (wrong subtype,
normal/contradicting result, or someone else's history). Teaches the
relevance-checking pattern the error analysis showed was missing from
the original training set, which only had generic "no mention at all"
examples for the Unsupported class.
"""

import pandas as pd
import random

random.seed(99)

def diabetes_hardneg():
    return random.choice([
        f"Type 1 diabetes diagnosed in childhood, on insulin pump, no renal involvement reported.",
        f"Gestational diabetes during pregnancy {random.randint(1,8)} years ago, resolved postpartum.",
        f"Diabetic nephropathy ruled out; recent eGFR {random.randint(70,100)}, no proteinuria on UA.",
        f"Strong family history of type 2 diabetes; patient's own glucose tolerance test normal.",
    ])

def chf_hardneg():
    return random.choice([
        f"Preserved ejection fraction {random.randint(55,65)}%, no heart failure symptoms, cardiac exam unremarkable.",
        f"History of peripartum cardiomyopathy, fully resolved, EF normalized to {random.randint(55,65)}%.",
        f"Family history of cardiomyopathy; patient's own echo and BNP within normal limits.",
    ])

def copd_hardneg():
    return random.choice([
        "Asthma diagnosed in childhood, well controlled on inhaled steroids, no COPD diagnosis on chart.",
        f"Spirometry within normal limits, FEV1 {random.randint(85,100)}% predicted, no obstructive pattern.",
        "Occupational dust exposure history noted; pulmonary function testing not yet performed.",
    ])

def ckd_hardneg():
    return random.choice([
        "Kidney transplant recipient, stable graft function, creatinine within normal limits.",
        f"CKD stage 2 noted on labs, eGFR {random.randint(65,85)}, routine monitoring only, no dialysis.",
        "Family history of polycystic kidney disease; patient's own renal function normal.",
    ])

def stroke_hardneg():
    return random.choice([
        "Migraine with aura, fully resolved, no infarct seen on imaging.",
        "Family history of stroke, patient has no personal history of cerebrovascular events.",
        "TIA symptoms resolved within minutes, full neurologic workup negative for infarct.",
    ])

def dementia_hardneg():
    return random.choice([
        f"Mild cognitive impairment ruled out on formal testing, MMSE {random.randint(28,30)}/30, normal for age.",
        "Delirium during recent hospitalization, fully resolved after treatment, baseline cognition intact.",
        "Family history of Alzheimer's disease; patient's own cognitive screening normal.",
    ])

def ra_hardneg():
    return random.choice([
        "Osteoarthritis of the right knee, mild, managed with acetaminophen as needed.",
        "RA serologies negative, no synovitis on exam, joint complaints attributed to overuse.",
        "Family history of rheumatoid arthritis; patient's own rheumatology workup unremarkable.",
    ])

generators = {
    "SP_DIABETES": diabetes_hardneg,
    "SP_CHF": chf_hardneg,
    "SP_COPD": copd_hardneg,
    "SP_CHRNKIDN": ckd_hardneg,
    "SP_STRKETIA": stroke_hardneg,
    "SP_ALZHDMTA": dementia_hardneg,
    "SP_RA_OA": ra_hardneg,
}

billed = pd.read_csv("data/processed/billed_code_reference.csv")
existing = pd.read_csv("data/processed/synthetic_notes_labeled.csv")
seen_texts = set(existing["note_text"])

PER_CONDITION = 15  # ~15 unique hard negatives per condition, ~105 total

rows = []
for condition, gen_fn in generators.items():
    billed_row = billed[billed["condition_flag"] == condition].iloc[0]
    added, attempts = 0, 0
    while added < PER_CONDITION and attempts < 100:
        text = gen_fn()
        attempts += 1
        if text in seen_texts:
            continue
        seen_texts.add(text)
        rows.append({
            "DESYNPUF_ID": f"HARDNEG_{condition}_{added:03d}",
            "age": random.randint(65, 90),
            "sex": random.choice(["M", "F"]),
            "condition_flag": condition,
            "billed_icd10_code": billed_row["billed_icd10_code"],
            "billed_hcc": billed_row["billed_hcc"],
            "billed_raf_value": billed_row["billed_raf_value"],
            "note_text": text,
            "documentation_level": "hard_negative",
            "label": "Unsupported",
        })
        added += 1
    print(f"{condition}: added {added} hard-negative examples")

hard_negs = pd.DataFrame(rows)
combined = pd.concat([existing, hard_negs], ignore_index=True)
combined.to_csv("data/processed/synthetic_notes_labeled.csv", index=False)

print()
print(f"Total hard negatives added: {len(hard_negs)}")
print(f"New total training rows: {len(combined)}")
print()
print("New overall label distribution:")
print(combined["label"].value_counts())