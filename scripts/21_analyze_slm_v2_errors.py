"""
Confusion matrix and error breakdown for the fine-tuned SLM benchmark,
mirroring the Claude analysis for direct comparison.
"""

import pandas as pd

results = pd.read_csv("data/processed/slm_v2_benchmark_results.csv")
golden = pd.read_csv("data/processed/golden_eval_set.csv")

merged = results.merge(golden, left_on="index", right_index=True, suffixes=("", "_golden"))

print("===== CONFUSION MATRIX: Fine-tuned SLM (rows = true label, columns = predicted) =====")
print(pd.crosstab(merged["true_label"], merged["model_answer"]))
print()
print("===== ACCURACY BY TRUE LABEL =====")
print(merged.groupby("true_label")["match"].mean())
print()
print("===== 8 SAMPLE WRONG ANSWERS =====")
print(merged[merged["match"] == False][["note_text", "true_label", "model_answer"]].head(8).to_string(index=False))