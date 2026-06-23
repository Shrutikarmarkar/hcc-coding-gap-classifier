"""
Generates a short clinical documentation snippet for every (patient,
condition) pair, varying how specifically it documents the condition --
fully detailed, vague, or entirely absent -- and labels each one
accordingly. This is the actual training data for the classifier.
"""

import pandas as pd
import numpy as np

np.random.seed(42)

# Two phrasing variants per specificity level, per condition.
# "full" = proves the billed code's clinical detail.
# "vague" = mentions the condition but lacks the proving detail.
note_templates = {
    "SP_DIABETES": {
        "full": [
            "Type 2 diabetes mellitus with diabetic nephropathy; eGFR declining, managed with insulin and ACE inhibitor.",
            "Long-standing type 2 diabetes with diabetic kidney disease confirmed on labs, eGFR 45, proteinuria present.",
        ],
        "vague": [
            "Patient reports history of diabetes, currently on metformin.",
            "Diabetes, stable, continue current regimen.",
        ],
    },
    "SP_CHF": {
        "full": [
            "Congestive heart failure, NYHA class III, recent echo showing EF 30%, on diuretic therapy.",
            "Chronic systolic heart failure, ejection fraction 35%, follow-up for medication titration.",
        ],
        "vague": [
            "History of heart failure, doing okay.",
            "Cardiac issues noted, follow up as needed.",
        ],
    },
    "SP_COPD": {
        "full": [
            "COPD on home oxygen; recent PFTs show FEV1 45% predicted.",
            "COPD with frequent exacerbations, using albuterol and tiotropium inhalers.",
        ],
        "vague": [
            "Patient has breathing problems, smoker.",
            "COPD noted in history, no acute issues today.",
        ],
    },
    "SP_CHRNKIDN": {
        "full": [
            "Chronic kidney disease stage 5, eGFR 12, on hemodialysis three times weekly.",
            "End-stage renal disease, dialysis dependent, nephrology following closely.",
        ],
        "vague": [
            "Some kidney problems noted on labs.",
            "History of kidney disease, will monitor.",
        ],
    },
    "SP_STRKETIA": {
        "full": [
            "History of acute ischemic stroke with residual left-sided weakness, on anticoagulation.",
            "Cerebral infarction documented on prior CT; currently in PT for residual deficits.",
        ],
        "vague": [
            "Patient had a stroke in the past.",
            "History of cerebrovascular event, stable.",
        ],
    },
    "SP_ALZHDMTA": {
        "full": [
            "Alzheimer's dementia, moderate stage, MMSE 14/30, requires assistance with daily activities.",
            "Progressive cognitive decline consistent with Alzheimer's disease; family reports increased confusion.",
        ],
        "vague": [
            "Patient seems a bit forgetful lately.",
            "Some memory issues noted, monitor.",
        ],
    },
    "SP_RA_OA": {
        "full": [
            "Rheumatoid arthritis with active synovitis in bilateral hands; elevated RF and anti-CCP, on methotrexate.",
            "RA flare with morning stiffness over one hour, MCP joint swelling, rheumatology following.",
        ],
        "vague": [
            "Joint pain reported.",
            "History of arthritis, no specific complaints today.",
        ],
    },
}

absent_note_options = [
    "Patient presents for routine follow-up. Vitals stable. No new complaints.",
    "Routine visit, general check-in. No acute concerns today.",
]

# Probabilities for how the note ends up written: 50% full, 35% vague, 15% absent
level_probs = {"full": 0.50, "vague": 0.35, "absent": 0.15}
label_map = {"full": "Fully Supported", "vague": "Insufficient", "absent": "Unsupported"}

ben = pd.read_csv("data/raw/desynpuf/1-sample-10000/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv")
billed = pd.read_csv("data/processed/billed_code_reference.csv")

condition_flags = list(note_templates.keys())
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
                note_text = np.random.choice(absent_note_options)
            else:
                note_text = np.random.choice(note_templates[condition][level])

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
print()
print("Label distribution:")
print(dataset["label"].value_counts())
print()
print("Sample rows:")
print(dataset[["condition_flag", "note_text", "label"]].head(6).to_string(index=False))