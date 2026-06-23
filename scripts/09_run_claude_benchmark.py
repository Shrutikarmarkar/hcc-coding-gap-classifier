"""
Runs Claude Opus 4.7 (the expensive frontier baseline) against the full
146-example golden eval set. Records the model's answer, whether it
matched the true label, exact token-based cost, and latency for every
single call.
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Official Anthropic pricing for claude-opus-4-7, per token
INPUT_COST_PER_TOKEN = 5.00 / 1_000_000
OUTPUT_COST_PER_TOKEN = 25.00 / 1_000_000

golden = pd.read_csv("data/processed/golden_eval_set.csv")

def build_prompt(note_text, billed_code, billed_hcc):
    return f"""You are reviewing a Medicare clinical note for risk-adjustment documentation compliance.

Billed diagnosis code: {billed_code}
Billed risk-adjustment category: {billed_hcc}

Clinical note:
\"\"\"{note_text}\"\"\"

Does this clinical note's documentation sufficiently support the billed diagnosis/category above?

Respond with EXACTLY one of these three labels, nothing else:
Fully Supported
Insufficient
Unsupported
"""

results = []
for i, row in golden.iterrows():
    prompt = build_prompt(row["note_text"], row["billed_icd10_code"], row["billed_hcc"])

    start = time.time()
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start

    model_answer = response.content[0].text.strip()
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = (input_tokens * INPUT_COST_PER_TOKEN) + (output_tokens * OUTPUT_COST_PER_TOKEN)

    results.append({
        "index": i,
        "condition_flag": row["condition_flag"],
        "true_label": row["label"],
        "model_answer": model_answer,
        "match": model_answer == row["label"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
        "latency_sec": latency,
    })

    print(f"[{i+1}/{len(golden)}] true={row['label']:18s} pred={model_answer:18s} match={model_answer == row['label']}")

results_df = pd.DataFrame(results)
results_df.to_csv("data/processed/claude_opus_benchmark_results.csv", index=False)

accuracy = results_df["match"].mean()
total_cost = results_df["cost_usd"].sum()
avg_latency = results_df["latency_sec"].mean()
std_latency = results_df["latency_sec"].std()

print()
print("===== BENCHMARK SUMMARY: Claude Opus 4.7 =====")
print(f"Accuracy: {accuracy:.1%}  ({results_df['match'].sum()}/{len(results_df)} correct)")
print(f"Total cost for {len(results_df)} notes: ${total_cost:.4f}")
print(f"Average cost per note: ${total_cost/len(results_df):.6f}")
print(f"Average latency: {avg_latency:.2f} sec (std: {std_latency:.2f})")