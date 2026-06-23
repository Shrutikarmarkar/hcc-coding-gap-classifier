"""
Explores the structure of the CMS DE-SynPUF Beneficiary Summary File —
specifically the demographic fields and chronic condition flags we'll
need to build our ground-truth dataset.
"""

import pandas as pd

ben = pd.read_csv("data/raw/desynpuf/1-sample-10000/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv")

print("Shape (rows, columns):", ben.shape)
print()
print("All column names:")
print(list(ben.columns))
print()
print("First 3 rows:")
print(ben.head(3))

print(ben['SP_DIABETES'].value_counts())
print(ben['BENE_SEX_IDENT_CD'].value_counts())

condition_flags = [
    'SP_ALZHDMTA', 'SP_CHF', 'SP_CHRNKIDN', 'SP_CNCR', 'SP_COPD',
    'SP_DEPRESSN', 'SP_DIABETES', 'SP_ISCHMCHT', 'SP_OSTEOPRS',
    'SP_RA_OA', 'SP_STRKETIA'
]

# Convert CMS's 1=Yes/2=No convention into a clean boolean
for col in condition_flags:
    ben[col] = ben[col] == 1

# Convert sex code into a readable label
ben['sex'] = ben['BENE_SEX_IDENT_CD'].map({1: 'M', 2: 'F'})

# Sanity check: counts should match what we already saw (3,794 True for diabetes)
print(ben['SP_DIABETES'].value_counts())
print(ben['sex'].value_counts())