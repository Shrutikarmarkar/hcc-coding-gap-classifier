"""
Defines the ONE code that gets billed for each condition (this determines
the dollar value actually claimed). Note specificity -- whether the
documentation actually SUPPORTS that billed code -- gets decided later,
when we write the note text itself in Phase 2.
"""

import pandas as pd

crosswalk = pd.read_csv("data/processed/icd10_to_hcc_v28_crosswalk.csv", dtype=str)
raf_table = pd.read_csv("data/processed/hcc_v28_raf_value_table.csv", dtype={"hcc_code": str})

billed_codes = {
    "SP_DIABETES":  "E1122",
    "SP_CHF":       "I509",
    "SP_COPD":      "J449",
    "SP_CHRNKIDN":  "N185",
    "SP_STRKETIA":  "I639",
    "SP_ALZHDMTA":  "G309",
    "SP_RA_OA":     "M069",
}

def lookup(code):
    match = crosswalk[crosswalk["diagnosis_code"] == code]
    if len(match) == 0:
        return None, 0.0
    hcc = match.iloc[0]["hcc_code"]
    hcc_full = hcc if hcc.startswith("HCC") else f"HCC{hcc}"
    val_row = raf_table[raf_table["hcc_code"] == hcc_full]
    return hcc_full, (val_row["raf_weight_cna"].values[0] if len(val_row) else 0.0)

rows = []
for condition, code in billed_codes.items():
    hcc, raf = lookup(code)
    rows.append({"condition_flag": condition, "billed_icd10_code": code, "billed_hcc": hcc, "billed_raf_value": raf})

reference = pd.DataFrame(rows)
reference.to_csv("data/processed/billed_code_reference.csv", index=False)
print(reference.to_string(index=False))