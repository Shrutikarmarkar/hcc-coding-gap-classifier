"""
Pulls the official CMS-HCC V28 ICD-10-to-HCC crosswalk, RAF dollar-value
coefficients, and HCC labels out of the hccinfhir package, and saves
clean, filtered CSVs into data/processed/.
"""

import pandas as pd
import hccinfhir
import os

# Find where pip installed the package's bundled data files
pkg_data_dir = os.path.join(os.path.dirname(hccinfhir.__file__), "data")

# Load the raw bundled files (these contain ALL model versions mixed together)
dx_to_cc = pd.read_csv(os.path.join(pkg_data_dir, "ra_dx_to_cc_2026.csv"))
coeffs = pd.read_csv(os.path.join(pkg_data_dir, "ra_coefficients_2026.csv"))
labels = pd.read_csv(os.path.join(pkg_data_dir, "ra_labels_2026.csv"))

# --- Filter down to ONLY the current V28 model ---
v28_dx_to_cc = dx_to_cc[dx_to_cc["model_name"] == "CMS-HCC Model V28"].copy()
v28_dx_to_cc.rename(columns={"cc": "hcc_code"}, inplace=True)

v28_labels = labels[labels["model_version"] == "V28"].copy()
v28_labels.rename(columns={"cc": "hcc_code"}, inplace=True)

# CNA = "Community NonDual Aged" segment -- the standard segment used
# for typical/average RAF reporting
coeffs_v28 = coeffs[(coeffs["model_domain"] == "CMS-HCC") & (coeffs["model_version"] == "C28")].copy()
cna = coeffs_v28[coeffs_v28["coefficient"].str.match(r"^CNA_HCC\d+$", na=False)].copy()
cna["hcc_code"] = "HCC" + cna["coefficient"].str.extract(r"CNA_HCC(\d+)")[0]
cna = cna[["hcc_code", "value"]].rename(columns={"value": "raf_weight_cna"})

# --- Build final clean output tables ---
crosswalk = v28_dx_to_cc[["diagnosis_code", "hcc_code"]]
raf_value_table = cna.merge(
    v28_labels[["hcc_code", "label"]], on="hcc_code", how="left"
).sort_values("raf_weight_cna", ascending=False)

# --- Save ---
os.makedirs("data/processed", exist_ok=True)
crosswalk.to_csv("data/processed/icd10_to_hcc_v28_crosswalk.csv", index=False)
raf_value_table.to_csv("data/processed/hcc_v28_raf_value_table.csv", index=False)

print(f"Crosswalk saved: {len(crosswalk)} ICD-10 codes mapped to HCCs")
print(f"RAF value table saved: {len(raf_value_table)} HCC categories")
print()
print("Top 5 highest-value HCCs:")
print(raf_value_table.head(5).to_string(index=False))