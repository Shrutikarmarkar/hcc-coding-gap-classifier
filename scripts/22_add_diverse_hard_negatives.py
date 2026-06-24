"""
Round 2 of hard-negative examples: genuinely diverse phrasing (not just
randomized numbers in the same sentence shape) covering the same three
failure patterns the error analysis revealed -- wrong subtype, normal/
resolved value, and family-history-not-patient. Checked against BOTH
the training set and the golden eval set to avoid any duplication.
"""

import pandas as pd
import random

random.seed(123)

def diabetes_neg():
    age = random.randint(8, 25)
    yrs = random.randint(2, 10)
    a1c = round(random.uniform(5.8, 6.8), 1)
    return random.choice([
        f"Type 1 diabetes mellitus, diagnosed at age {age}, no diabetic complications screened to date.",
        "Steroid-induced hyperglycemia during recent prednisone course, resolved after taper.",
        f"History of gestational diabetes {yrs} years ago, normal glucose tolerance since delivery.",
        "MODY (maturity-onset diabetes of the young) diagnosed via genetic testing, managed with diet alone.",
        f"T2DM with A1c {a1c}% (well controlled), no nephropathy, retinopathy, or neuropathy on screening.",
        f"Diabetes in remission per recent labs, A1c {a1c}%, no medications currently required.",
        "Strong family history of type 2 diabetes in both parents; patient's own fasting glucose normal.",
        "Sibling recently diagnosed with diabetes; patient screened negative on oral glucose tolerance test.",
    ])

def chf_neg():
    ef = random.randint(55, 65)
    yrs = random.randint(2, 12)
    return random.choice([
        f"Hypertensive heart disease without heart failure; LVH on echo, EF preserved at {ef}%.",
        "History of mild valvular regurgitation, asymptomatic, no heart failure diagnosis.",
        f"Cardiomyopathy of pregnancy, fully resolved postpartum {yrs} years ago.",
        f"EF {ef}% on most recent echo, NYHA class I, no current heart failure symptoms.",
        f"Prior heart failure diagnosis from {yrs} years ago, now resolved, off all cardiac medications.",
        "Father had heart failure; patient's own cardiac workup including echo and BNP unremarkable.",
    ])

def copd_neg():
    fev1 = random.randint(85, 100)
    yrs = random.randint(3, 20)
    return random.choice([
        "Asthma since childhood, well controlled, no COPD or smoking history.",
        "Bronchiectasis on imaging, not COPD; pulmonology following separately.",
        "Restrictive lung disease from prior radiation, distinct from obstructive COPD pattern.",
        f"Spirometry within normal limits, FEV1 {fev1}% predicted, no obstructive disease.",
        f"Former smoker, quit {yrs} years ago, lungs clear, no COPD diagnosis established.",
        "Mother had COPD; patient is a lifelong non-smoker with normal pulmonary function.",
    ])

def ckd_neg():
    yrs = random.randint(2, 15)
    cr = round(random.uniform(0.8, 1.2), 1)
    egfr = random.randint(75, 100)
    return random.choice([
        "Acute kidney injury post-surgery, fully resolved, baseline renal function normal.",
        "Nephrolithiasis (kidney stones) only, no chronic kidney disease present.",
        f"Single kidney post-nephrectomy {yrs} years ago, remaining kidney function normal.",
        f"Kidney transplant {yrs} years ago, excellent graft function, creatinine {cr}, no rejection.",
        f"CKD stage 1 only, eGFR {egfr}, monitored annually, no intervention needed.",
        "Family history of polycystic kidney disease; patient's own renal ultrasound and labs normal.",
    ])

def stroke_neg():
    age = random.randint(55, 75)
    yrs = random.randint(1, 8)
    return random.choice([
        "Complex migraine with neurological aura, mimicking stroke symptoms, MRI negative for infarct.",
        "Transient global amnesia episode, fully resolved, distinct from cerebrovascular event.",
        "Syncope workup negative for stroke; cardiac etiology identified instead.",
        f"TIA {yrs} years ago, full neurologic recovery, no residual deficits, MRI clear.",
        "Carotid stenosis monitored, asymptomatic, no history of actual stroke or infarct.",
        f"Mother had a stroke at age {age}; patient's own neurologic exam and imaging normal.",
    ])

def dementia_neg():
    mmse = random.randint(27, 30)
    return random.choice([
        f"Mild cognitive impairment only, does not meet criteria for dementia, MMSE {mmse}/30.",
        "Depression-related pseudodementia, cognition improved after treatment of mood disorder.",
        "Delirium during hospitalization, fully resolved, baseline cognition intact at discharge.",
        f"Cognitive screening MMSE {mmse}/30, within normal range for age and education.",
        "Prior concern for memory loss, formal neuropsych testing entirely normal.",
        "Strong family history of Alzheimer's disease; patient's own cognitive testing unremarkable.",
    ])

def ra_neg():
    das = round(random.uniform(1.5, 2.5), 1)
    yrs = random.randint(1, 6)
    return random.choice([
        "Osteoarthritis confirmed on imaging, RF and anti-CCP negative, not rheumatoid arthritis.",
        "Gout flare in big toe, uric acid elevated, distinct from rheumatoid arthritis.",
        "Fibromyalgia diagnosis, diffuse pain without joint swelling or inflammatory markers.",
        f"RA in clinical remission, DAS28 {das}, off DMARDs for {yrs} years.",
        "Joint pain workup negative for inflammatory arthritis; RF and anti-CCP both negative.",
        "Mother has rheumatoid arthritis; patient's own joint exam and serologies normal.",
    ])

generators = {
    "SP_DIABETES": diabetes_neg,
    "SP_CHF": chf_neg,
    "SP_COPD": copd_neg,
    "SP_CHRNKIDN": ckd_neg,
    "SP_STRKETIA": stroke_neg,
    "SP_ALZHDMTA": dementia_neg,
    "SP_RA_OA": ra_neg,
}

billed = pd.read_csv("data/processed/billed_code_reference.csv")
existing = pd.read_csv("data/processed/synthetic_notes_labeled.csv")
golden = pd.read_csv("data/processed/golden_eval_set.csv")

seen_texts = set(existing["note_text"]) | set(golden["note_text"])

PER_CONDITION = 15

rows = []
for condition, gen_fn in generators.items():
    billed_row = billed[billed["condition_flag"] == condition].iloc[0]
    added, attempts = 0, 0
    while added < PER_CONDITION and attempts < 150:
        text = gen_fn()
        attempts += 1
        if text in seen_texts:
            continue
        seen_texts.add(text)
        rows.append({
            "DESYNPUF_ID": f"HARDNEG2_{condition}_{added:03d}",
            "age": random.randint(60, 90),
            "sex": random.choice(["M", "F"]),
            "condition_flag": condition,
            "billed_icd10_code": billed_row["billed_icd10_code"],
            "billed_hcc": billed_row["billed_hcc"],
            "billed_raf_value": billed_row["billed_raf_value"],
            "note_text": text,
            "documentation_level": "hard_negative_v2",
            "label": "Unsupported",
        })
        added += 1
    print(f"{condition}: added {added} diverse hard-negative examples (after {attempts} attempts)")

hard_negs_v2 = pd.DataFrame(rows)

OVERSAMPLE_FACTOR = 4
oversampled = pd.concat([hard_negs_v2] * OVERSAMPLE_FACTOR, ignore_index=True)

combined = pd.concat([existing, oversampled], ignore_index=True)
combined.to_csv("data/processed/synthetic_notes_labeled.csv", index=False)

print()
print(f"New unique diverse hard negatives: {len(hard_negs_v2)}")
print(f"After {OVERSAMPLE_FACTOR}x oversampling: {len(oversampled)} rows")
print(f"New total training rows: {len(combined)}")
print()
print("New overall label distribution:")
print(combined["label"].value_counts())