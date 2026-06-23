"""
Converts the labeled training notes into the JSONL format mlx-lm expects
for fine-tuning, splitting by UNIQUE NOTE TEXT into train/valid sets so
no exact sentence appears on both sides. The golden eval set is left
completely untouched -- it remains the final held-out test, used later.
"""

import pandas as pd
import json
import os
import random

random.seed(42)

df = pd.read_csv("data/processed/synthetic_notes_labeled.csv")

def build_prompt(note_text, billed_code, billed_hcc):
    return f"""You are reviewing a Medicare clinical note for risk-adjustment documentation compliance.

Billed diagnosis code: {billed_code}
Billed risk-adjustment category: {billed_hcc}

Clinical note:
\"\"\"{note_text}\"\"\"

Classify the documentation using EXACTLY one of these three labels:

Fully Supported - The note explicitly documents the specific clinical detail
(lab value, severity, complication, staging) required to justify this exact
billed category, not just the general disease.

Insufficient - The note mentions the condition or related symptoms, but
lacks the specific clinical detail needed to justify this billed category.
The condition is present in the note, just not proven with enough detail.

Unsupported - The note does not mention this condition at all, describes a
different/unrelated condition, or only mentions it in a context that doesn't
apply to the patient (e.g. family history, a resolved past condition, or a
different subtype of the disease).

Respond with EXACTLY one of these three labels, nothing else."""

# Split by UNIQUE note text first -- this is the leak-prevention step
# Map each unique text to its label (every unique text has exactly one
# consistent label, since it was generated at a fixed specificity level)
text_to_label = df.drop_duplicates("note_text").set_index("note_text")["label"].to_dict()

train_texts, valid_texts = set(), set()

for label in df["label"].unique():
    texts_for_label = [t for t, l in text_to_label.items() if l == label]
    random.shuffle(texts_for_label)

    n_valid = max(1, round(len(texts_for_label) * 0.15)) if len(texts_for_label) >= 2 else 0
    valid_texts.update(texts_for_label[:n_valid])
    train_texts.update(texts_for_label[n_valid:])

    print(f"{label}: {len(texts_for_label)} unique texts -> {len(texts_for_label)-n_valid} train / {n_valid} valid")

# Safety check: confirm zero overlap
assert len(train_texts & valid_texts) == 0, "LEAK DETECTED: overlap between train/valid texts"

train_rows = df[df["note_text"].isin(train_texts)]
valid_rows = df[df["note_text"].isin(valid_texts)]

print(f"Unique note texts: {len(train_texts) + len(valid_texts)} total -> {len(train_texts)} train / {len(valid_texts)} valid")
print(f"Actual rows: {len(train_rows)} train / {len(valid_rows)} valid (out of {len(df)} total)")
print()
print("Train label distribution:")
print(train_rows["label"].value_counts())
print()
print("Valid label distribution:")
print(valid_rows["label"].value_counts())

def write_jsonl(rows_df, path):
    with open(path, "w") as f:
        for _, row in rows_df.iterrows():
            prompt = build_prompt(row["note_text"], row["billed_icd10_code"], row["billed_hcc"])
            f.write(json.dumps({"prompt": prompt, "completion": row["label"]}) + "\n")

os.makedirs("data/finetune", exist_ok=True)
write_jsonl(train_rows, "data/finetune/train.jsonl")
write_jsonl(valid_rows, "data/finetune/valid.jsonl")

print()
print("Saved data/finetune/train.jsonl and data/finetune/valid.jsonl")