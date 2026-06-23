"""
Scans every CHF note in both the training set and golden eval set for
language describing an ACUTE presentation (admission, exacerbation,
decompensation) -- which the billed category HCC226 explicitly excludes.
Flags any "Fully Supported" CHF note that may actually be mislabeled.
"""

import pandas as pd

acute_keywords = ["admit", "hospitaliz", "exacerbation", "decompensat"]

def flag_acute(text):
    text_lower = text.lower()
    return any(kw in text_lower for kw in acute_keywords)

training = pd.read_csv("data/processed/synthetic_notes_labeled.csv")
golden = pd.read_csv("data/processed/golden_eval_set.csv")

chf_train = training[training["condition_flag"] == "SP_CHF"].copy()
chf_golden = golden[golden["condition_flag"] == "SP_CHF"].copy()

chf_train["likely_acute"] = chf_train["note_text"].apply(flag_acute)
chf_golden["likely_acute"] = chf_golden["note_text"].apply(flag_acute)

print("===== TRAINING SET: CHF notes labeled 'Fully Supported' that mention acute language =====")
flagged_train = chf_train[(chf_train["label"] == "Fully Supported") & (chf_train["likely_acute"])]
print(f"Flagged: {len(flagged_train)} out of {len(chf_train[chf_train['label']=='Fully Supported'])} 'Fully Supported' CHF rows")
print(flagged_train["note_text"].drop_duplicates().to_string(index=False))
print()

print("===== GOLDEN EVAL SET: CHF notes labeled 'Fully Supported' that mention acute language =====")
flagged_golden = chf_golden[(chf_golden["label"] == "Fully Supported") & (chf_golden["likely_acute"])]
print(f"Flagged: {len(flagged_golden)} out of {len(chf_golden[chf_golden['label']=='Fully Supported'])} 'Fully Supported' CHF rows")
print(flagged_golden[["note_text"]].to_string(index=False))