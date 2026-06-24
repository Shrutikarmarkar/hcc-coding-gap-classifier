"""
Runs the fine-tuned Phi-3.5-mini (LoRA adapter, iteration 100 checkpoint)
against the same 146-example golden eval set Claude Opus 4.7 was tested
on. Records accuracy, latency, and a confusion matrix -- same format as
the Claude benchmark, for a direct side-by-side comparison.
"""

import time
import pandas as pd
from mlx_lm import load, generate

MODEL_PATH = "mlx-community/Phi-3.5-mini-instruct-4bit"
ADAPTER_PATH = "adapters/hcc-classifier-best-v2"

print("Loading model + adapter...")
model, tokenizer = load(MODEL_PATH, adapter_path=ADAPTER_PATH)

golden = pd.read_csv("data/processed/golden_eval_set.csv")

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

results = []
for i, row in golden.iterrows():
    raw_prompt = build_prompt(row["note_text"], row["billed_icd10_code"], row["billed_hcc"])
    messages = [{"role": "user", "content": raw_prompt}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    start = time.time()
    response = generate(model, tokenizer, prompt=prompt, max_tokens=8, verbose=False)
    if i < 5:
        print(f"  RAW OUTPUT: {repr(response)}")
    latency = time.time() - start

    model_answer = response.split("<|")[0].strip()

    results.append({
        "index": i,
        "condition_flag": row["condition_flag"],
        "true_label": row["label"],
        "model_answer": model_answer,
        "match": model_answer == row["label"],
        "latency_sec": latency,
    })

    print(f"[{i+1}/{len(golden)}] true={row['label']:18s} pred={model_answer:18s} match={model_answer == row['label']}")

results_df = pd.DataFrame(results)
results_df.to_csv("data/processed/slm_v2_benchmark_results.csv", index=False)

accuracy = results_df["match"].mean()
avg_latency = results_df["latency_sec"].mean()
std_latency = results_df["latency_sec"].std()

print()
print("===== BENCHMARK SUMMARY: Fine-tuned Phi-3.5-mini (LoRA) =====")
print(f"Accuracy: {accuracy:.1%}  ({results_df['match'].sum()}/{len(results_df)} correct)")
print(f"Cost per note: $0.00 (runs locally, no API charges)")
print(f"Average latency: {avg_latency:.2f} sec (std: {std_latency:.2f})")