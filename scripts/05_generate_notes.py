"""
Generates a short clinical documentation snippet for every (patient,
condition) pair, using randomized template functions so the same
specificity level produces many different unique sentences instead of
repeating a fixed handful -- avoiding a model that just memorizes strings.
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

def diabetes_note(level):
    if level == "full":
        egfr = random.randint(28, 55)
        skeleton = random.choice([
            f"Type 2 diabetes mellitus with diabetic nephropathy; eGFR {egfr}, managed with insulin and ACE inhibitor.",
            f"Long-standing type 2 diabetes with diabetic kidney disease confirmed on labs, eGFR {egfr}, proteinuria present.",
        ])
        return skeleton
    else:
        med = random.choice(["metformin", "glipizide", "insulin", "sitagliptin"])
        return random.choice([
            f"Patient reports history of diabetes, currently on {med}.",
            f"Diabetes, stable, continue {med}.",
        ])

def chf_note(level):
    if level == "full":
        ef = random.randint(20, 38)
        nyha = random.choice(["II", "III", "IV"])
        return random.choice([
            f"Congestive heart failure, NYHA class {nyha}, recent echo showing EF {ef}%, on diuretic therapy.",
            f"Chronic systolic heart failure, ejection fraction {ef}%, follow-up for medication titration.",
        ])
    else:
        return random.choice([
            "History of heart failure, doing okay.",
            "Cardiac issues noted, follow up as needed.",
        ])

def copd_note(level):
    if level == "full":
        fev1 = random.randint(30, 55)
        return random.choice([
            f"COPD on home oxygen; recent PFTs show FEV1 {fev1}% predicted.",
            f"COPD with frequent exacerbations, FEV1 {fev1}% predicted, using albuterol and tiotropium inhalers.",
        ])
    else:
        return random.choice([
            "Patient has breathing problems, smoker.",
            "COPD noted in history, no acute issues today.",
        ])

def ckd_note(level):
    if level == "full":
        egfr = random.randint(8, 14)
        return random.choice([
            f"Chronic kidney disease stage 5, eGFR {egfr}, on hemodialysis three times weekly.",
            f"End-stage renal disease, eGFR {egfr}, dialysis dependent, nephrology following closely.",
        ])
    else:
        return random.choice([
            "Some kidney problems noted on labs.",
            "History of kidney disease, will monitor.",
        ])

def stroke_note(level):
    if level == "full":
        side = random.choice(["left-sided", "right-sided"])
        months = random.randint(2, 18)
        return random.choice([
            f"History of acute ischemic stroke {months} months ago with residual {side} weakness, on anticoagulation.",
            f"Cerebral infarction documented on prior CT {months} months ago; currently in PT for residual {side} deficits.",
        ])
    else:
        return random.choice([
            "Patient had a stroke in the past.",
            "History of cerebrovascular event, stable.",
        ])

def dementia_note(level):
    if level == "full":
        mmse = random.randint(10, 19)
        return random.choice([
            f"Alzheimer's dementia, moderate stage, MMSE {mmse}/30, requires assistance with daily activities.",
            f"Progressive cognitive decline consistent with Alzheimer's disease, MMSE {mmse}/30; family reports increased confusion.",
        ])
    else:
        return random.choice([
            "Patient seems a bit forgetful lately.",
            "Some memory issues noted, monitor.",
        ])

def ra_note(level):
    if level == "full":
        joints = random.randint(3, 9)
        return random.choice([
            f"Rheumatoid arthritis with active synovitis in {joints} joints; elevated RF and anti-CCP, on methotrexate.",
            f"RA flare with morning stiffness over one hour, {joints} swollen joints, rheumatology following.",
        ])
    else:
        return random.choice([
            "Joint pain reported.",
            "History of arthritis, no specific complaints today.",
        ])

note_generators = {
    "SP_DIABETES": diabetes_note,
    "SP_CHF": chf_note,
    "SP_COPD": copd_note,
    "SP_CHRNKIDN": ckd_note,
    "SP_STRKETIA": stroke_note,
    "SP_ALZHDMTA": dementia_note,
    "SP_RA_OA": ra_note,
}

absent_note_options = [
    "Patient presents for routine follow-up. Vitals stable. No new complaints.",
    "Routine visit, general check-in. No acute concerns today.",
    "Annual wellness visit, no acute issues identified today.",
]

level_probs = {"full": 0.50, "vague": 0.35, "absent": 0.15}
label_map = {"full": "Fully Supported", "vague": "Insufficient", "absent": "Unsupported"}

ben = pd.read_csv("data/raw/desynpuf/1-sample-10000/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv")
billed = pd.read_csv("data/processed/billed_code_reference.csv")

condition_flags = list(note_generators.keys())
for col in condition_flags:
    ben[col] = ben[col] == 1
ben["sex"] = ben["BENE_SEX_IDENT_CD"].map({1: "M", 2: "F"})
ben["age"] = 2008 - ben["BENE_BIRTH_DT"].astype(str).str[:4].astype(int)

rows = []
for _, row in ben.iterrows():
    for condition in condition_flags:
        if row[condition]:
            level = np.random.choice(list(level_probs.keys()), p=list(level_probs.values()))
            if level == "absent":
                note_text = random.choice(absent_note_options)
            else:
                note_text = note_generators[condition](level)

            billed_row = billed[billed["condition_flag"] == condition].iloc[0]
            rows.append({
                "DESYNPUF_ID": row["DESYNPUF_ID"],
                "age": row["age"],
                "sex": row["sex"],
                "condition_flag": condition,
                "billed_icd10_code": billed_row["billed_icd10_code"],
                "billed_hcc": billed_row["billed_hcc"],
                "billed_raf_value": billed_row["billed_raf_value"],
                "note_text": note_text,
                "documentation_level": level,
                "label": label_map[level],
            })

dataset = pd.DataFrame(rows)
dataset.to_csv("data/processed/synthetic_notes_labeled.csv", index=False)

print(f"Total labeled note records: {len(dataset)}")
print(f"Unique note texts: {dataset['note_text'].nunique()}")
print()
print("Label distribution:")
print(dataset["label"].value_counts())
print()
print("Sample rows:")
print(dataset[["condition_flag", "note_text", "label"]].head(8).to_string(index=False))