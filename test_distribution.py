import pandas as pd
import openpyxl
import os
#
# radiomics_dir = "/gpfs/home6/palfken/radiomics_features.csv"
#
# uncertainty_dir = "/gpfs/home6/palfken/unc_map_features.csv"
#
#
# unc_df = pd.read_csv(uncertainty_dir)
# rad_df = pd.read_csv(radiomics_dir)
#
#
# # --- Step 1: Create a global 'empty_bool' and delete all '*empty_flag' columns ---
#
# unc_df.rename(columns={'confidence_empty_flag': 'empty_bool'}, inplace=True)
#
# # Drop all columns ending with 'empty_flag'
# empty_flag_cols = [col for col in unc_df.columns if col.endswith('empty_flag')]
# unc_df.drop(columns=empty_flag_cols, inplace=True)
#
# # --- Step 2: Keep only rows in rad_df with case_ids present in unc_df ---
# # Assuming the column with case ids is called 'case_id' in both
# rad_df = rad_df[rad_df['case_id'].isin(unc_df['case_id'])].copy()
#
# # --- Step 3: Merge the two dataframes on 'case_id' and remove duplicate case_id column ---
# merged_df = pd.merge(rad_df, unc_df, on='case_id', how='inner')
#
# # If the merge somehow created a second case_id column (e.g., 'case_id_y'), drop it
# merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
#
# merged_df.to_csv("/gpfs/home6/palfken/final_features.csv")
#
#
# print(f'Number of samples in final df: {len(merged_df)}. Should match old df :{len(unc_df)}')
# print(merged_df.head(20))
# print(merged_df.columns)
test_dir = "/scratch/bmep/plalfken/asqac-stbs JOB.xlsx"
subtypes_dir = "/scratch/bmep/plalfken/accession numbers.xlsx"
test_data_dir = "/scratch/bmep/plalfken/test_data_nifti_1"


# Read your files
test_df = pd.read_excel(test_dir)

subtype_df = pd.read_excel(subtypes_dir)
print(subtype_df.head())
print(test_df.head())


# Strip all column names
test_df.columns = test_df.columns.str.strip()
subtype_df.columns = subtype_df.columns.str.strip()

print("Test columns:", test_df.columns)
print("Subtype columns:", subtype_df.columns)

# Now merge safely
matching_subtypes = subtype_df[subtype_df['MDN'].isin(test_df['MDN'])]
matching_subtypes = matching_subtypes[['MDN','tumor_class']]

merged = test_df.merge(
    matching_subtypes,
    left_on='MDN',
    right_on='MDN',
    how='left'
).drop(columns=['MDN'])



# --- filter to only cases that exist as NIfTI files ---
# Normalize accession numbers to match NIfTI filenames
accnr_col = 'new_accnr'
merged[accnr_col] = merged[accnr_col].str.replace('-', '_')

test_files = [f for f in os.listdir(test_data_dir) if f.endswith('.nii.gz')]

present_case_ids = set()
for f in test_files:
    if f.endswith('_0000.nii.gz'):
        case_id = f.rsplit('_0000.nii.gz', 1)[0]  # split from the right
        present_case_ids.add(case_id)
    else:
        # fallback if naming is unexpected
        present_case_ids.add(f.replace('.nii.gz', ''))

print(present_case_ids)

merged[accnr_col] = merged[accnr_col].astype(str)
test_df_filtered = merged[merged[accnr_col].isin(present_case_ids)].copy()

print(test_df_filtered.head(50))
print(len(test_df_filtered))

counts = test_df_filtered['tumor_class'].value_counts()
print(f'Tumor counts: {counts}')

# First, make a copy of 'accnr' with '-' replaced by '_'
test_df_filtered['new_accnr'] = test_df_filtered['new_accnr'].str.replace('-', '_')

# Define your in-distribution tumor classes
id_tumors = ['WM Leiomyosarcoom','WM Liposarcomen','Desmoid fibromatose/ aggressieve fibromatose'
]

# Create a new column 'dist' to mark in-distribution (ID) vs out-of-distribution (OOD)
test_df_filtered['dist'] = test_df_filtered['tumor_class'].apply(lambda x: 'ID' if x in id_tumors else 'OOD')

# Check
print(test_df_filtered[['new_accnr', 'tumor_class', 'dist']].head(20))



print(test_df_filtered['new_accnr'].unique)
old_dir = "/scratch/bmep/plalfken/test_table.csv"
old_df = pd.read_csv(old_dir)
print(old_df['new_accnr'].unique)
final_df = pd.concat([old_df, test_df_filtered],ignore_index=True)
final_df.to_csv("/scratch/bmep/plalfken/test_table1.csv", index=False)

check = "/scratch/bmep/plalfken/test_table1.csv"
df = pd.read_csv(check)

print(df.head(20))
print(df[df['dist'] == 'ID'])
print(len(df[df['dist'] == 'ID']))

print(len(df))