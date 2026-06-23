"""
Hand-written golden evaluation set -- deliberately different phrasing
and structure from the training templates, including a few "trap" cases
that test genuine reading comprehension rather than keyword-matching.
This set is held out and NEVER used for training.
"""

import pandas as pd

golden_examples = [
    # SP_DIABETES (billed: E1122 / HCC37)
    ("SP_DIABETES", "DM2 w/ nephropathy noted on last labs (Cr 2.1, eGFR 29); continues lisinopril for renal protection.", "Fully Supported"),
    ("SP_DIABETES", "Diabetic kidney disease confirmed, referred to nephrology, A1c 8.2, microalbuminuria present.", "Fully Supported"),
    ("SP_DIABETES", "Sugar levels a little high, will recheck next visit.", "Insufficient"),
    ("SP_DIABETES", "Hx of DM, no complaints today.", "Insufficient"),
    ("SP_DIABETES", "Pt here for med refill, no new issues.", "Unsupported"),
    ("SP_DIABETES", "Brittle type 1 diabetes since childhood, frequent hypoglycemic episodes, on insulin pump.", "Unsupported"),  # TRAP: wrong diabetes type, doesn't support billed code

    # SP_CHF (billed: I509 / HCC226)
    ("SP_CHF", "Echo last month showed EF of 28%, discussed worsening systolic function and diuretic adjustment.", "Fully Supported"),
    ("SP_CHF", "Admitted last week for heart failure exacerbation, now on optimized GDMT, NYHA III symptoms persist.", "Fully Supported"),
    ("SP_CHF", "Heart's been a little weak, watching it.", "Insufficient"),
    ("SP_CHF", "Pt mentions some fatigue, hx of cardiac issues.", "Insufficient"),
    ("SP_CHF", "Annual physical, unremarkable.", "Unsupported"),
    ("SP_CHF", "Preserved EF 55% on recent echo, mild diastolic dysfunction only.", "Insufficient"),  # TRAP: normal EF, weak support

    # SP_COPD (billed: J449 / HCC280)
    ("SP_COPD", "Spirometry confirms severe obstruction, FEV1/FVC ratio reduced, on triple inhaler therapy.", "Fully Supported"),
    ("SP_COPD", "COPD exacerbation x2 this year requiring steroids, baseline O2 sat 88% on room air.", "Fully Supported"),
    ("SP_COPD", "Smoker, occasional cough.", "Insufficient"),
    ("SP_COPD", "Lung issues, nothing acute.", "Insufficient"),
    ("SP_COPD", "Routine BP check, all stable.", "Unsupported"),
    ("SP_COPD", "Uses home oxygen 2L NC, prior PFTs unavailable.", "Insufficient"),  # gray area: suggestive but no objective data

    # SP_CHRNKIDN (billed: N185 / HCC326)
    ("SP_CHRNKIDN", "Dialysis 3x/week per renal team, access via AV fistula, last K+ 4.8.", "Fully Supported"),
    ("SP_CHRNKIDN", "Creatinine continues to climb, GFR under 15, transplant workup initiated.", "Fully Supported"),
    ("SP_CHRNKIDN", "Renal function not great, will keep an eye on it.", "Insufficient"),
    ("SP_CHRNKIDN", "Pt's kidneys have been an issue per old records.", "Insufficient"),
    ("SP_CHRNKIDN", "Follow up visit, denies new symptoms.", "Unsupported"),
    ("SP_CHRNKIDN", "Kidney transplant recipient, stable graft function, creatinine within normal limits.", "Unsupported"),  # TRAP: normal function, doesn't support stage 5 billing

    # SP_STRKETIA (billed: I639 / HCC249)
    ("SP_STRKETIA", "S/p CVA 6 months ago, residual right-sided weakness, continues clopidogrel.", "Fully Supported"),
    ("SP_STRKETIA", "MRI confirmed infarct in MCA territory, now in outpatient PT for gait training.", "Fully Supported"),
    ("SP_STRKETIA", "Pt mentions a stroke a while back, seems fine now.", "Insufficient"),
    ("SP_STRKETIA", "Old neuro event per chart, no current deficits noted.", "Insufficient"),
    ("SP_STRKETIA", "Here for flu shot, no concerns.", "Unsupported"),
    ("SP_STRKETIA", "Family history of stroke, patient has no personal history of cerebrovascular events.", "Unsupported"),  # TRAP: family history, not patient's own diagnosis

    # SP_ALZHDMTA (billed: G309 / HCC127)
    ("SP_ALZHDMTA", "MMSE scored 12 today, family reports increasing reliance on caregiver for ADLs.", "Fully Supported"),
    ("SP_ALZHDMTA", "Progressive memory loss consistent with AD, now requires supervision at home.", "Fully Supported"),
    ("SP_ALZHDMTA", "A little more forgetful than usual per spouse.", "Insufficient"),
    ("SP_ALZHDMTA", "Pt seems okay today, family didn't raise concerns.", "Insufficient"),
    ("SP_ALZHDMTA", "Routine eye exam referral, no other issues.", "Unsupported"),

    # SP_RA_OA (billed: M069 / HCC93)
    ("SP_RA_OA", "Positive RF/anti-CCP, synovitis in MCPs bilaterally, increasing methotrexate dose.", "Fully Supported"),
    ("SP_RA_OA", "Rheum following for RA, morning stiffness over one hour, new hand swelling noted.", "Fully Supported"),
    ("SP_RA_OA", "Some joint aches, otherwise fine.", "Insufficient"),
    ("SP_RA_OA", "Hx of arthritis, no flare today.", "Insufficient"),
    ("SP_RA_OA", "Pt in for vaccine, no complaints.", "Unsupported"),
    ("SP_RA_OA", "Osteoarthritis of the right knee, mild, managed with acetaminophen as needed.", "Unsupported"),  # TRAP: OA not RA, doesn't support RA billing
]

golden_df = pd.DataFrame(golden_examples, columns=["condition_flag", "note_text", "label"])

billed = pd.read_csv("data/processed/billed_code_reference.csv")
golden_df = golden_df.merge(billed, on="condition_flag", how="left")

golden_df.to_csv("data/processed/golden_eval_set.csv", index=False)

print(f"Total golden eval examples: {len(golden_df)}")
print()
print("Label distribution:")
print(golden_df["label"].value_counts())
print()
print("Per-condition counts:")
print(golden_df["condition_flag"].value_counts())