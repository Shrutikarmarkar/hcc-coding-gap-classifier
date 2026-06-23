"""
Corrects the 3 golden eval CHF examples that describe an acute
presentation but were billed against the chronic/unspecified code
(I509/HCC226). The note supports SOME heart failure billing, just not
specifically this one -- which is the definition of "Insufficient,"
not "Fully Supported."
"""

import pandas as pd

acute_keywords = ["admit", "hospitaliz", "exacerbation", "decompensat"]

def flag_acute(text):
    return any(kw in text.lower() for kw in acute_keywords)

golden = pd.read_csv("data/processed/golden_eval_set.csv")

mask = (
    (golden["condition_flag"] == "SP_CHF") &
    (golden["label"] == "Fully Supported") &
    (golden["note_text"].apply(flag_acute))
)

print(f"Relabeling {mask.sum()} rows from 'Fully Supported' to 'Insufficient':")
print(golden.loc[mask, "note_text"].to_string(index=False))

golden.loc[mask, "label"] = "Insufficient"
golden.to_csv("data/processed/golden_eval_set.csv", index=False)

print()
print("Updated label distribution:")
print(golden["label"].value_counts())