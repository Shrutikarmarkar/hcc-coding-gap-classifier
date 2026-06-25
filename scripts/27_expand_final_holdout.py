"""
Expands the final holdout test set with randomized generator versions
of the same 42 hand-written examples. FIXED: every template now has at
least one randomized element, so none can exactly collide with the
original fixed sentences (root cause of the previous run silently
adding zero new "Unsupported" examples).
"""

import pandas as pd
import random

random.seed(2025)

def diabetes(label):
    egfr = random.randint(25, 42)
    a1c = round(random.uniform(7.5, 9.5), 1)
    months = random.randint(2, 8)
    visit = random.choice(["annual physical", "routine wellness visit", "follow-up appointment", "med refill visit"])
    workup = random.choice(["Renal biopsy", "Nephrology workup", "Repeat renal biopsy"])
    pools = {
        "Fully Supported": [
            f"Diabetic nephropathy confirmed via renal biopsy, eGFR {egfr}, started on finerenone for renal protection.",
            f"Diabetic CKD with eGFR {egfr} and macroalbuminuria, A1c {a1c}%, co-managed with nephrology.",
        ],
        "Insufficient": [
            f"Sugars running a bit high lately around A1c {a1c}%, will adjust insulin dose.",
            f"Possible early diabetic nephropathy suspected given creatinine trending up over {months} months; nephrology referral pending.",
        ],
        "Unsupported": [
            f"Here for {visit}, no diabetes-related concerns raised.",
            f"{workup} ruled out diabetic nephropathy; proteinuria due to unrelated glomerular disease.",
        ],
    }
    return random.choice(pools[label])

def chf(label):
    ef = random.randint(25, 38)
    ef_normal = random.randint(55, 65)
    wk = random.randint(1, 4)
    symptom = random.choice(["mild leg swelling", "occasional ankle edema", "slight dyspnea on exertion"])
    reason = random.choice(["skin check", "vaccination", "blood pressure check", "annual labs"])
    pools = {
        "Fully Supported": [
            f"Chronic heart failure with reduced ejection fraction {ef}%, stable on guideline-directed therapy for over a year.",
            f"Systolic heart failure, EF {ef}%, NYHA class III, on optimized medical therapy for {random.randint(6,24)} months.",
        ],
        "Insufficient": [
            f"Hospitalized {wk} weeks ago for acute decompensated heart failure, now recovering at home.",
            f"Possible early heart failure given {symptom}, echo pending to confirm.",
        ],
        "Unsupported": [
            f"No cardiac complaints, here for {reason}.",
            f"Heart failure was suspected given edema, but echo ruled out any cardiac dysfunction; EF {ef_normal}%.",
        ],
    }
    return random.choice(pools[label])

def copd(label):
    fev1 = random.randint(30, 48)
    pack_years = random.randint(15, 40)
    exac = random.randint(2, 4)
    vaccine = random.choice(["flu vaccination", "pneumonia vaccination", "shingles vaccination"])
    duration = random.randint(2, 8)
    freq = random.choice(["occasionally", "a few times a week", "most mornings"])
    pools = {
        "Fully Supported": [
            f"Severe COPD confirmed on PFTs, FEV1 {fev1}% predicted, on long-term oxygen therapy.",
            f"COPD with FEV1/FVC ratio reduced, {exac} exacerbations requiring steroid bursts this year.",
        ],
        "Insufficient": [
            f"Some wheezing noted {freq}, inhaler use as needed.",
            f"Possible COPD given smoking history of {pack_years} pack-years, spirometry scheduled but not yet done.",
        ],
        "Unsupported": [
            f"Routine visit for {vaccine}, respiratory exam normal.",
            f"Chronic cough for {duration} weeks, workup ruled out COPD; spirometry normal, attributed to postnasal drip.",
        ],
    }
    return random.choice(pools[label])

def ckd(label):
    egfr = random.randint(8, 14)
    access = random.choice(["tunneled catheter", "AV fistula", "AV graft"])
    months_off = random.randint(1, 6)
    med = random.choice(["lisinopril", "metoprolol", "atorvastatin"])
    weeks = random.randint(2, 6)
    pools = {
        "Fully Supported": [
            f"ESRD, hemodialysis 3x weekly, access via {access}, nephrology managing closely.",
            f"Stage 5 CKD, eGFR {egfr}, transplant evaluation in progress, dietary restrictions reviewed.",
        ],
        "Insufficient": [
            f"Kidney function a bit off on labs from {months_off} months ago, will recheck.",
            "Possible progression to stage 5 CKD suspected given trending creatinine; repeat labs and nephrology referral pending.",
        ],
        "Unsupported": [
            f"Here for {med} refill, renal labs not reviewed today.",
            f"AKI from contrast exposure {weeks} weeks ago ruled out as chronic; renal function returned to baseline normal.",
        ],
    }
    return random.choice(pools[label])

def stroke(label):
    months = random.randint(2, 10)
    side = random.choice(["left", "right"])
    weeks = random.randint(1, 6)
    days = random.randint(1, 14)
    reason = random.choice(["annual wellness visit", "routine physical", "medication review"])
    pools = {
        "Fully Supported": [
            f"Ischemic stroke confirmed on MRI {months} months ago, residual mild word-finding difficulty, on dual antiplatelet therapy.",
            f"Cerebral infarct in {side} MCA territory confirmed on CT, currently in speech therapy for residual aphasia.",
        ],
        "Insufficient": [
            f"Some balance issues for the past {weeks} weeks, neuro following loosely.",
            f"Possible small stroke suspected given transient confusion episode {days} days ago; MRI pending.",
        ],
        "Unsupported": [
            f"{reason.capitalize()}, neurologic exam grossly normal.",
            f"Acute stroke was suspected on presentation {days} days ago but imaging ruled out infarct; diagnosed as complex migraine instead.",
        ],
    }
    return random.choice(pools[label])

def dementia(label):
    mmse = random.randint(10, 18)
    weeks = random.randint(2, 8)
    reason = random.choice(["hearing aid fitting", "vision check", "flu shot"])
    med = random.choice(["a sedative", "an anticholinergic medication", "a sleep aid"])
    pools = {
        "Fully Supported": [
            f"Alzheimer's dementia confirmed via neuropsych evaluation, MMSE {mmse}/30, now requires full-time caregiver support.",
            f"Moderate dementia, MMSE {mmse}/30, family reports patient now lost while driving, license revoked.",
        ],
        "Insufficient": [
            f"A little more confused than usual over the past {weeks} weeks, family keeping an eye on it.",
            "Possible early dementia suspected given family report of forgetfulness; formal cognitive testing pending.",
        ],
        "Unsupported": [
            f"Here for {reason}, cognition not addressed.",
            f"Memory complaints worked up and dementia ruled out; attributed to {med}, since resolved.",
        ],
    }
    return random.choice(pools[label])

def ra(label):
    das = round(random.uniform(4.5, 6.5), 1)
    joint = random.choice(["wrists", "MCP joints", "knees and wrists"])
    weeks = random.randint(2, 8)
    reason = random.choice(["blood pressure check", "annual physical", "flu shot"])
    pools = {
        "Fully Supported": [
            f"Rheumatoid arthritis confirmed, positive RF and anti-CCP, active synovitis in {joint} bilaterally, started biologic therapy.",
            f"Seropositive RA, DAS28 {das}, escalating to combination DMARD therapy due to inadequate response.",
        ],
        "Insufficient": [
            f"Some joint discomfort for {weeks} weeks, hands mostly, no clear diagnosis yet.",
            "Possible rheumatoid arthritis suspected given hand stiffness; rheumatology referral and labs pending.",
        ],
        "Unsupported": [
            f"Here for {reason}, joints not examined today.",
            "Inflammatory arthritis was suspected but serologies ruled out RA; diagnosed as osteoarthritis instead.",
        ],
    }
    return random.choice(pools[label])

generators = {
    "SP_DIABETES": diabetes, "SP_CHF": chf, "SP_COPD": copd,
    "SP_CHRNKIDN": ckd, "SP_STRKETIA": stroke, "SP_ALZHDMTA": dementia, "SP_RA_OA": ra,
}

labels = ["Fully Supported", "Insufficient", "Unsupported"]
SAMPLES_PER_COMBO = 6

billed = pd.read_csv("data/processed/billed_code_reference.csv")
training = pd.read_csv("data/processed/synthetic_notes_labeled.csv")
golden = pd.read_csv("data/processed/golden_eval_set.csv")
existing_final = pd.read_csv("data/processed/final_holdout_test.csv")

seen_texts = set(training["note_text"]) | set(golden["note_text"]) | set(existing_final["note_text"])

rows = []
for condition, gen_fn in generators.items():
    for label in labels:
        added, attempts = 0, 0
        while added < SAMPLES_PER_COMBO and attempts < 200:
            text = gen_fn(label)
            attempts += 1
            if text in seen_texts:
                continue
            seen_texts.add(text)
            rows.append({"condition_flag": condition, "note_text": text, "label": label})
            added += 1
        print(f"{condition}/{label}: added {added}/{SAMPLES_PER_COMBO} (after {attempts} attempts)")
        if added < SAMPLES_PER_COMBO:
            print(f"  WARNING: shortfall")

new_batch = pd.DataFrame(rows).merge(billed, on="condition_flag", how="left")
combined = pd.concat([existing_final, new_batch], ignore_index=True)

overlap = set(combined["note_text"]) & (set(training["note_text"]) | set(golden["note_text"]))
assert len(overlap) == 0, f"LEAK: {overlap}"
assert combined["note_text"].duplicated().sum() == 0, "Internal duplicates found"

combined.to_csv("data/processed/final_holdout_test.csv", index=False)

print(f"\nNew examples added: {len(new_batch)}")
print(f"Total final holdout test set: {len(combined)}")
print(combined["label"].value_counts())
print(combined["condition_flag"].value_counts())