"""
Joins our billed HCC codes against the official label table to show the
actual category descriptions -- including any hidden exclusions in the
wording (like "Except Acute") that our note text might violate.
"""

import pandas as pd

billed = pd.read_csv("data/processed/billed_code_reference.csv", dtype={"billed_hcc": str})
raf_table = pd.read_csv("data/processed/hcc_v28_raf_value_table.csv", dtype={"hcc_code": str})

merged = billed.merge(raf_table[["hcc_code", "label"]], left_on="billed_hcc", right_on="hcc_code", how="left")

print(merged[["condition_flag", "billed_icd10_code", "billed_hcc", "label", "billed_raf_value"]].to_string(index=False))