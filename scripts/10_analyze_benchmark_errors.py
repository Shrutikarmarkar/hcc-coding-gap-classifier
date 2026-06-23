"""
Builds a confusion matrix from the Claude benchmark results to diagnose
WHERE the errors are concentrated -- random mistakes, or a systematic
confusion between two specific labels.
"""

import pandas as pd

results = pd.read_csv("data/processed/claude_opus_benchmark_results.csv")
golden = pd.read_csv("data/processed/golden_eval_set.csv")

merged = results.merge(golden, left_on="index", right_index=True, suffixes=("", "_golden"))

print("===== CONFUSION MATRIX (rows = true label, columns = Claude's answer) =====")
confusion = pd.crosstab(merged["true_label"], merged["model_answer"])
print(confusion)
print()

print("===== ACCURACY BY TRUE LABEL =====")
print(merged.groupby("true_label")["match"].mean())
print()

print("===== 8 SAMPLE WRONG ANSWERS =====")
wrong = merged[merged["match"] == False][["note_text", "true_label", "model_answer"]]
print(wrong.head(8).to_string(index=False))