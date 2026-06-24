"""
Isolates performance on the 5 deliberately hand-designed "trap" cases --
notes with specific clinical detail that's actually irrelevant, wrong-
subtype, or contradicting the billed code -- comparing Claude vs the
fine-tuned SLM specifically on these, since this is the real test of
reasoning vs. keyword-matching.
"""

import pandas as pd

trap_note_texts = [
    "Brittle type 1 diabetes since childhood, frequent hypoglycemic episodes, on insulin pump.",
    "Preserved EF 55% on recent echo, mild diastolic dysfunction only.",
    "Kidney transplant recipient, stable graft function, creatinine within normal limits.",
    "Family history of stroke, patient has no personal history of cerebrovascular events.",
    "Osteoarthritis of the right knee, mild, managed with acetaminophen as needed.",
]

golden = pd.read_csv("data/processed/golden_eval_set.csv")
claude_results = pd.read_csv("data/processed/claude_opus_benchmark_results.csv")
slm_results = pd.read_csv("data/processed/slm_benchmark_results.csv")

trap_indices = golden[golden["note_text"].isin(trap_note_texts)].index.tolist()

claude_trap = claude_results[claude_results["index"].isin(trap_indices)]
slm_trap = slm_results[slm_results["index"].isin(trap_indices)]

print(f"Found {len(trap_indices)} trap cases in golden eval set")
print()
print(f"Claude Opus 4.7 trap-case accuracy: {claude_trap['match'].mean():.1%} ({claude_trap['match'].sum()}/{len(claude_trap)})")
print(f"Fine-tuned SLM trap-case accuracy:  {slm_trap['match'].mean():.1%} ({slm_trap['match'].sum()}/{len(slm_trap)})")
print()

comparison = golden.loc[trap_indices, ["note_text", "label"]].copy()
comparison["claude_answer"] = claude_results.set_index("index").loc[trap_indices, "model_answer"].values
comparison["slm_answer"] = slm_results.set_index("index").loc[trap_indices, "model_answer"].values
print(comparison.to_string(index=False))