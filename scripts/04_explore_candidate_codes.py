"""
Tests multiple real candidate ICD-10 codes per condition against our
verified V28 crosswalk, so we can see which ones actually produce
DIFFERENT HCC outcomes -- instead of assuming.
"""

import pandas as pd

crosswalk = pd.read_csv("data/processed/icd10_to_hcc_v28_crosswalk.csv", dtype=str)
raf_table = pd.read_csv("data/processed/hcc_v28_raf_value_table.csv", dtype={"hcc_code": str})

def lookup(code):
    match = crosswalk[crosswalk["diagnosis_code"] == code]
    if len(match) == 0:
        return "NO HCC", 0.0
    hcc = match.iloc[0]["hcc_code"]
    hcc_full = hcc if hcc.startswith("HCC") else f"HCC{hcc}"
    val_row = raf_table[raf_table["hcc_code"] == hcc_full]
    raf_value = val_row["raf_weight_cna"].values[0] if len(val_row) else 0.0
    return hcc_full, raf_value

# Multiple real candidates per condition -- we're exploring, not assuming
candidates = {
    "COPD":        ["J449", "J440", "J441"],
    "Stroke":      ["I639", "I679", "I693"],
    "Dementia":    ["G309", "F0280", "F0281"],
    "RA/OA":       ["M069", "M1990", "M0600"],
}

for condition, codes in candidates.items():
    print(f"--- {condition} ---")
    for code in codes:
        hcc, raf = lookup(code)
        print(f"  {code:8s} -> {hcc:10s} raf={raf}")
    print()