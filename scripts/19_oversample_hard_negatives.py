"""
Duplicates the hard-negative examples several times each, so they get
meaningful representation during training instead of being drowned out
by the much larger original dataset (71 out of 13,624 rows is only
0.5% -- too thin a signal for the model to reliably learn from).
"""

import pandas as pd

OVERSAMPLE_FACTOR = 6

df = pd.read_csv("data/processed/synthetic_notes_labeled.csv")

hard_negs = df[df["documentation_level"] == "hard_negative"]
everything_else = df[df["documentation_level"] != "hard_negative"]

oversampled_hard_negs = pd.concat([hard_negs] * OVERSAMPLE_FACTOR, ignore_index=True)

combined = pd.concat([everything_else, oversampled_hard_negs], ignore_index=True)
combined.to_csv("data/processed/synthetic_notes_labeled.csv", index=False)

print(f"Hard negatives: {len(hard_negs)} unique -> {len(oversampled_hard_negs)} rows after {OVERSAMPLE_FACTOR}x oversampling")
print(f"New total training rows: {len(combined)}")
print()
print("New overall label distribution:")
print(combined["label"].value_counts())