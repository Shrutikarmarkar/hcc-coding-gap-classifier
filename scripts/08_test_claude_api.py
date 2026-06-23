'''
What this script does: loads your API key quietly from .env, grabs just the first example from your golden eval set, builds a prompt that gives Claude the clinical note plus the billed code, asks it to pick one of your three labels, and checks whether it matches the correct answer you already wrote down
'''
import os
import pandas as pd
from dotenv import load_dotenv
import anthropic

load_dotenv()  # reads ANTHROPIC_API_KEY out of your .env file

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

golden = pd.read_csv("data/processed/golden_eval_set.csv")
example = golden.iloc[0]

note_text = example["note_text"]
billed_code = example["billed_icd10_code"]
billed_hcc = example["billed_hcc"]
true_label = example["label"]

prompt = f"""You are reviewing a Medicare clinical note for risk-adjustment documentation compliance.

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

response = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=20,
    messages=[{"role": "user", "content": prompt}],
)

model_answer = response.content[0].text.strip()

print("Note text:", note_text)
print("Billed code / HCC:", billed_code, "/", billed_hcc)
print("True label:", true_label)
print("Claude's answer:", model_answer)
print("Match:", model_answer == true_label)


'''
We're using claude-opus-4-7 here deliberately — it's Anthropic's strongest, most expensive model, which is exactly what we want playing the role of the "expensive frontier baseline" our whole project is benchmarking against.
'''

