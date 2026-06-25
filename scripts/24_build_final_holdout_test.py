"""
FINAL HELD-OUT TEST SET. Written once, inspected for results exactly
once, after the model is already locked in. Do NOT use this set's
results to make any further training decisions -- if you do, it stops
being a valid measure of real-world performance.
"""

import pandas as pd

final_test = [
    # SP_DIABETES (billed E1122 / HCC37)
    ("SP_DIABETES", "Diabetic nephropathy confirmed via renal biopsy, now started on finerenone for renal protection.", "Fully Supported"),
    ("SP_DIABETES", "Diabetic CKD with eGFR 38 and macroalbuminuria, co-managed with nephrology.", "Fully Supported"),
    ("SP_DIABETES", "Sugars running a bit high lately, will adjust insulin dose.", "Insufficient"),
    ("SP_DIABETES", "Possible early diabetic nephropathy suspected given rising creatinine; nephrology referral pending.", "Insufficient"),
    ("SP_DIABETES", "Here for annual physical, no diabetes-related concerns raised.", "Unsupported"),
    ("SP_DIABETES", "Renal biopsy ruled out diabetic nephropathy; proteinuria due to unrelated glomerular disease.", "Unsupported"),

    # SP_CHF (billed I509 / HCC226 -- remember the acute exclusion)
    ("SP_CHF", "Chronic heart failure with reduced ejection fraction 32%, stable on guideline-directed therapy for over a year.", "Fully Supported"),
    ("SP_CHF", "Hospitalized two weeks ago for acute decompensated heart failure, now recovering at home.", "Insufficient"),
    ("SP_CHF", "Some shortness of breath with exertion, cardiology aware.", "Insufficient"),
    ("SP_CHF", "Possible early heart failure given mild leg swelling, echo pending to confirm.", "Insufficient"),
    ("SP_CHF", "No cardiac complaints, here for skin check.", "Unsupported"),
    ("SP_CHF", "Heart failure was suspected given edema, but echo ruled out any cardiac dysfunction; EF normal.", "Unsupported"),

    # SP_COPD (billed J449 / HCC280)
    ("SP_COPD", "Severe COPD confirmed on PFTs, FEV1 38% predicted, on long-term oxygen therapy.", "Fully Supported"),
    ("SP_COPD", "COPD with FEV1/FVC ratio 0.55, frequent exacerbations requiring steroid bursts twice this year.", "Fully Supported"),
    ("SP_COPD", "Some wheezing noted, inhaler use as needed.", "Insufficient"),
    ("SP_COPD", "Possible COPD given smoking history, spirometry scheduled but not yet done.", "Insufficient"),
    ("SP_COPD", "Routine visit for vaccination, respiratory exam normal.", "Unsupported"),
    ("SP_COPD", "Chronic cough workup ruled out COPD; spirometry normal, attributed to postnasal drip.", "Unsupported"),

    # SP_CHRNKIDN (billed N185 / HCC326)
    ("SP_CHRNKIDN", "ESRD, hemodialysis 3x weekly, access via tunneled catheter, nephrology managing closely.", "Fully Supported"),
    ("SP_CHRNKIDN", "Stage 5 CKD, eGFR 11, transplant evaluation in progress, dietary restrictions reviewed.", "Fully Supported"),
    ("SP_CHRNKIDN", "Kidney function a bit off on last labs, will recheck.", "Insufficient"),
    ("SP_CHRNKIDN", "Possible progression to stage 5 CKD suspected given trending creatinine; repeat labs and nephrology referral pending.", "Insufficient"),
    ("SP_CHRNKIDN", "Here for med refill, renal labs not reviewed today.", "Unsupported"),
    ("SP_CHRNKIDN", "AKI from contrast exposure ruled out as chronic; renal function returned to baseline normal.", "Unsupported"),

    # SP_STRKETIA (billed I639 / HCC249)
    ("SP_STRKETIA", "Ischemic stroke confirmed on MRI three months ago, residual mild word-finding difficulty, on dual antiplatelet therapy.", "Fully Supported"),
    ("SP_STRKETIA", "Cerebral infarct in left MCA territory confirmed on CT, currently in speech therapy for residual aphasia.", "Fully Supported"),
    ("SP_STRKETIA", "Some balance issues lately, neuro following loosely.", "Insufficient"),
    ("SP_STRKETIA", "Possible small stroke suspected given transient confusion episode; MRI pending.", "Insufficient"),
    ("SP_STRKETIA", "Annual wellness visit, neurologic exam grossly normal.", "Unsupported"),
    ("SP_STRKETIA", "Acute stroke was suspected on presentation but imaging ruled out infarct; diagnosed as complex migraine instead.", "Unsupported"),

    # SP_ALZHDMTA (billed G309 / HCC127)
    ("SP_ALZHDMTA", "Alzheimer's dementia confirmed via neuropsych evaluation, MMSE 11/30, now requires full-time caregiver support.", "Fully Supported"),
    ("SP_ALZHDMTA", "Moderate dementia, MMSE 14/30, family reports patient now lost while driving, license revoked.", "Fully Supported"),
    ("SP_ALZHDMTA", "A little more confused than usual, family keeping an eye on it.", "Insufficient"),
    ("SP_ALZHDMTA", "Possible early dementia suspected given family report of forgetfulness; formal cognitive testing pending.", "Insufficient"),
    ("SP_ALZHDMTA", "Here for hearing aid fitting, cognition not addressed.", "Unsupported"),
    ("SP_ALZHDMTA", "Memory complaints worked up and dementia ruled out; attributed to medication side effect, since resolved.", "Unsupported"),

    # SP_RA_OA (billed M069 / HCC93)
    ("SP_RA_OA", "Rheumatoid arthritis confirmed, positive RF and anti-CCP, active synovitis in wrists bilaterally, started biologic therapy.", "Fully Supported"),
    ("SP_RA_OA", "Seropositive RA, DAS28 5.8, escalating to combination DMARD therapy due to inadequate response.", "Fully Supported"),
    ("SP_RA_OA", "Some joint discomfort, hands mostly, no clear diagnosis yet.", "Insufficient"),
    ("SP_RA_OA", "Possible rheumatoid arthritis suspected given hand stiffness; rheumatology referral and labs pending.", "Insufficient"),
    ("SP_RA_OA", "Here for blood pressure check, joints not examined today.", "Unsupported"),
    ("SP_RA_OA", "Inflammatory arthritis was suspected but serologies ruled out RA; diagnosed as osteoarthritis instead.", "Unsupported"),
]

final_df = pd.DataFrame(final_test, columns=["condition_flag", "note_text", "label"])

billed = pd.read_csv("data/processed/billed_code_reference.csv")
final_df = final_df.merge(billed, on="condition_flag", how="left")

# Safety check: confirm zero overlap with training data or the development (golden) set
training = pd.read_csv("data/processed/synthetic_notes_labeled.csv")
golden = pd.read_csv("data/processed/golden_eval_set.csv")
overlap = set(final_df["note_text"]) & (set(training["note_text"]) | set(golden["note_text"]))
assert len(overlap) == 0, f"LEAK: {overlap}"

final_df.to_csv("data/processed/final_holdout_test.csv", index=False)

print(f"Final holdout test set: {len(final_df)} examples")
print(final_df["label"].value_counts())
print(final_df["condition_flag"].value_counts())
print()
print("Zero overlap with training or development set: CONFIRMED")