"""
Expands the golden evaluation set with a second, larger batch of
hand-designed examples -- written in a different style/vocabulary from
both the training templates (script 05) and the original 41 examples
(script 06), to avoid any phrasing overlap. Combines with the existing
golden set and saves the full expanded version.
"""

import pandas as pd
import random

random.seed(7)

def diabetes(label):
    wk = random.randint(1,8)
    pools = {
        "Fully Supported": [
            f"DM2 c/b diabetic nephropathy; recent Cr {round(random.uniform(1.6,2.8),1)}, microalbuminuria confirmed on UA.",
            f"Type 2 DM with documented diabetic kidney disease, nephrology co-managing, urine ACR elevated at {random.randint(300,900)}.",
        ],
        "Insufficient": [
            f"Diabetic, labs pending x{wk} weeks, continue current meds.",
            f"T2DM noted in PMH for {random.randint(2,15)} years, no specific complications documented this visit.",
        ],
        "Unsupported": [
            f"Pt denies any diabetes history; here for unrelated knee pain x{wk} weeks.",
            f"Gestational diabetes resolved {random.randint(1,10)} years after delivery, no current diabetic diagnosis.",
        ],
    }
    return random.choice(pools[label])

def chf(label):
    wk = random.randint(1,8)
    pools = {
        "Fully Supported": [
            f"Echo {random.randint(1,6)} months ago revealed EF {random.randint(20,35)}%; pt symptomatic with class III heart failure on exam.",
            f"Decompensated heart failure requiring hospitalization {random.randint(1,4)} weeks ago; BNP elevated at admission, now euvolemic.",
        ],
        "Insufficient": [
            f"Cardiac hx significant x{random.randint(1,10)} years, no acute distress today.",
            f"Hx of heart issues, ambulates without difficulty over past {wk} weeks.",
        ],
        "Unsupported": [
            f"Pt here for routine labs x{wk} weeks f/u, cardiovascular exam unremarkable.",
            f"EF {random.randint(55,65)}% on recent echo, no heart failure symptoms reported.",
        ],
    }
    return random.choice(pools[label])

def copd(label):
    wk = random.randint(1,8)
    pools = {
        "Fully Supported": [
            f"Severe COPD, FEV1 {random.randint(28,48)}% of predicted, on supplemental O2 at home.",
            f"Documented COPD GOLD stage {random.randint(3,4)}, frequent rescue inhaler use, baseline dyspnea on exertion.",
        ],
        "Insufficient": [
            f"Hx of lung disease x{random.randint(1,15)} years, breathing okay today.",
            f"Former smoker, mild cough x{wk} weeks, no further workup.",
        ],
        "Unsupported": [
            f"Pt presents for vaccination, lungs clear, no respiratory complaints x{wk} weeks.",
            f"Asthma well controlled x{random.randint(1,10)} years, no COPD diagnosis on chart.",
        ],
    }
    return random.choice(pools[label])

def ckd(label):
    wk = random.randint(1,8)
    pools = {
        "Fully Supported": [
            f"ESRD on dialysis via AV graft, last session {random.randint(1,3)} days ago, K+ within range.",
            f"Stage 5 CKD confirmed, GFR {random.randint(8,14)}, transplant eval underway.",
        ],
        "Insufficient": [
            f"Renal labs slightly abnormal x{wk} weeks, will repeat next visit.",
            f"Hx of kidney trouble per old chart from {random.randint(1,8)} years back, no current workup.",
        ],
        "Unsupported": [
            f"Pt seen for med refill x{wk} weeks, renal function not addressed.",
            f"S/p kidney transplant {random.randint(1,10)} years ago, graft functioning well, creatinine normal.",
        ],
    }
    return random.choice(pools[label])

def stroke(label):
    wk = random.randint(1,8)
    pools = {
        "Fully Supported": [
            f"CVA {random.randint(2,14)} months ago, residual {'left' if random.random()<0.5 else 'right'} hemiparesis, continues PT/OT, on antiplatelet therapy.",
            f"Acute infarct confirmed on imaging, NIHSS {random.randint(3,9)} at presentation, now stable on secondary prevention.",
        ],
        "Insufficient": [
            f"Hx of stroke per chart from {random.randint(1,10)} years ago, no current deficits noted.",
            f"Pt mentions a 'mini-stroke' {random.randint(2,12)} years ago, unclear records.",
        ],
        "Unsupported": [
            f"TIA ruled out after workup {wk} weeks ago, no stroke diagnosis confirmed.",
            f"Routine visit, neuro exam grossly intact x{wk} weeks, no stroke history elicited.",
        ],
    }
    return random.choice(pools[label])

def dementia(label):
    wk = random.randint(1,8)
    pools = {
        "Fully Supported": [
            f"AD diagnosis confirmed via neuropsych testing, MMSE {random.randint(10,18)}, family providing 24/7 supervision.",
            f"Moderate dementia, MoCA {random.randint(8,15)}, increasing care needs noted by caregiver.",
        ],
        "Insufficient": [
            f"Family notes mild forgetfulness x{wk} weeks, otherwise functioning independently.",
            f"Cognitive complaints raised {random.randint(1,6)} months ago, formal testing not yet done.",
        ],
        "Unsupported": [
            f"Cognitive screen normal today, no concerns raised x{wk} weeks.",
            f"Pt's spouse has dementia diagnosed {random.randint(1,8)} years ago; patient himself is cognitively intact.",
        ],
    }
    return random.choice(pools[label])

def ra(label):
    wk = random.randint(1,8)
    pools = {
        "Fully Supported": [
            f"Active RA, DAS28 score {round(random.uniform(4.0,6.5),1)}, on combination DMARD therapy.",
            f"Seropositive RA with synovitis on exam x{wk} weeks, rheumatology adjusting biologics.",
        ],
        "Insufficient": [
            f"Joint stiffness reported x{wk} weeks, etiology unclear.",
            f"Hx of inflammatory arthritis from {random.randint(1,12)} years ago, currently asymptomatic.",
        ],
        "Unsupported": [
            f"Knee pain attributed to injury {wk} weeks ago, no inflammatory arthritis present.",
            f"OA of hips noted on imaging {random.randint(1,5)} years ago, RA serologies negative.",
        ],
    }
    return random.choice(pools[label])

generators = {
    "SP_DIABETES": diabetes,
    "SP_CHF": chf,
    "SP_COPD": copd,
    "SP_CHRNKIDN": ckd,
    "SP_STRKETIA": stroke,
    "SP_ALZHDMTA": dementia,
    "SP_RA_OA": ra,
}

labels = ["Fully Supported", "Insufficient", "Unsupported"]
SAMPLES_PER_COMBO = 5  # 7 conditions x 3 labels x 5 = 105 new examples

rows = []
for condition, gen_fn in generators.items():
    for label in labels:
        for _ in range(SAMPLES_PER_COMBO):
            rows.append({
                "condition_flag": condition,
                "note_text": gen_fn(label),
                "label": label,
            })

new_batch = pd.DataFrame(rows)

billed = pd.read_csv("data/processed/billed_code_reference.csv")
new_batch = new_batch.merge(billed, on="condition_flag", how="left")

existing = pd.read_csv("data/processed/golden_eval_set.csv")
seen_texts = set(existing["note_text"])

rows = []
for condition, gen_fn in generators.items():
    for label in labels:
        added = 0
        attempts = 0
        while added < SAMPLES_PER_COMBO and attempts < 50:
            text = gen_fn(label)
            attempts += 1
            if text in seen_texts:
                continue  # collision -- try again
            seen_texts.add(text)
            rows.append({"condition_flag": condition, "note_text": text, "label": label})
            added += 1
        if added < SAMPLES_PER_COMBO:
            print(f"WARNING: only generated {added}/{SAMPLES_PER_COMBO} unique examples for {condition}/{label}")

new_batch = pd.DataFrame(rows)
billed = pd.read_csv("data/processed/billed_code_reference.csv")
new_batch = new_batch.merge(billed, on="condition_flag", how="left")

combined = pd.concat([existing, new_batch], ignore_index=True)
duplicates = combined["note_text"].duplicated().sum()
combined.to_csv("data/processed/golden_eval_set.csv", index=False)

print(f"New examples added: {len(new_batch)}")
print(f"Total golden eval set size: {len(combined)}")
print(f"Exact duplicate note texts found: {duplicates}")
print()
print("Label distribution:")
print(combined["label"].value_counts())
print()
print("Per-condition counts:")
print(combined["condition_flag"].value_counts())