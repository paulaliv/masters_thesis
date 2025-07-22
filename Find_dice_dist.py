import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import shutil


df_dir = "/home/bmep/plalfken/my-scratch/nnUNet/Final_dice_dist.csv"
df_final = "/home/bmep/plalfken/my-scratch/nnUNet/Final_dice_dist1.csv"
data20 ="/scratch/bmep/plalfken/Dice_scores_20epochs.csv"
data5 = "/scratch/bmep/plalfken/Dice_scores_5epochs.csv"
remove_ids = [
    "DES_0010", "DES_0012", "DES_0035", "DES_0037", "DES_0046", "DES_0050",
    "DES_0059", "DES_0062", "DES_0075", "DES_0102", "DES_0123", "DES_0142",
    "DES_0189", "LIP_0013", "LIP_0020", "LIP_0022", "LIP_0027", "LIP_0046",
    "LIP_0069", "LIP_0083", "LIP_0087", "LIP_0115"
]
df5 = pd.read_csv(data5)
df20 = pd.read_csv(data20)
df30 = pd.read_csv(df_dir)
df_final = pd.read_csv(df_final)


# Remove them
df_final = df_final[~df_final['case_id'].isin(remove_ids)]

# Optional: Reset index after filtering
df= df_final.reset_index(drop=True)


print(f"Number of cases in df5: {len(df5)}")
print(f"Number of cases in df20: {len(df20)}")
print(f"Number of cases in df30: {len(df30)}")
print(f"Number of cases in df final: {len(df_final)}")




def base_case_id(x):
    if x.startswith("30EP_"):
        return x[5:]  # remove "20EP_" prefix
    if x.startswith("20EP_"):
        return x[5:]
    return x

df['base_case_id'] = df['case_id'].apply(base_case_id)

# Now prepare stratification labels on unique base cases:
# We want 1 label per base case for stratified splitting




dice_bins = [0, 0.1, 0.5, 0.7, 1]
dice_labels = ['Fail', 'Poor', 'Moderate', 'Good']

# Get one row per base_case_id (you can take the first occurrence)
base_cases = df.drop_duplicates(subset=['base_case_id']).copy()

print(base_cases.columns)

base_cases['dice_category'] = pd.cut(base_cases['dice_5'], bins=dice_bins, labels=dice_labels, include_lowest=True)

# Combine for stratification
base_cases['stratify_label'] = base_cases['tumor_class'].astype(str) + "_" + base_cases['dice_category'].astype(str)

# Add dice_category to the full df too:
df['dice_category'] = pd.cut(
    df['dice_5'], bins=dice_bins, labels=dice_labels, include_lowest=True
)


from sklearn.model_selection import StratifiedKFold


# Initialize StratifiedKFold with 5 splits, shuffle for randomness
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

splits = []

base_case_ids = base_cases['base_case_id'].values
stratify_labels = base_cases['stratify_label'].values

for train_idx, val_idx in skf.split(base_case_ids, stratify_labels):
    train_base = base_case_ids[train_idx]
    val_base = base_case_ids[val_idx]

    # Now assign ALL cases with these base_case_ids to train/val
    train_cases = df[df['base_case_id'].isin(train_base)]['case_id'].tolist()
    val_cases = df[df['base_case_id'].isin(val_base)]['case_id'].tolist()

    print(f"Train cases count: {len(train_cases)}")
    print(f"Val cases count: {len(val_cases)}")

    splits.append({'train': train_cases, 'val': val_cases})


for i, split in enumerate(splits):
    train_idx = df['case_id'].isin(split['train'])
    val_idx = df['case_id'].isin(split['val'])

    print(f"--- Split {i + 1} ---")
    print("Training set tumor_class distribution:")
    print(df.loc[train_idx, 'tumor_class'].value_counts())  # normalized freq
    print("Training set dice_category distribution:")
    print(df.loc[train_idx, 'dice_category'].value_counts())

    print("\nValidation set tumor_class distribution:")
    print(df.loc[val_idx, 'tumor_class'].value_counts())
    print("Validation set dice_category distribution:")
    print(df.loc[val_idx, 'dice_category'].value_counts())
    print("\n")

output_path = '/home/bmep/plalfken/my-scratch/nnUNet/Final_dice_dist1.csv'
output_dir = os.path.dirname(output_path)

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Now write the JSON file
with open(output_path, 'w') as f:
    json.dump(splits, f, indent=4)

# #
# image_dir = "/gpfs/home6/palfken/30QA_images/"
# dst_dir = "/gpfs/home6/palfken/QA_dataTr_final/"
# # Filter for case IDs starting with '30EP_'
# filtered_ids = df[df['case_id'].str.startswith('30EP_')]['case_id']
#
#
# base_ids = [cid.replace('30EP_', '') for cid in filtered_ids]
# file_endings = ['_img.npy', '_mask.npy', '_entropy.npy', '_confidence.npy', '_epkl.npy', '_mutual_info.npy']
#
# for base_id in base_ids:
#     for suffix in file_endings:
#         source_file = os.path.join(image_dir, f'20EP_{base_id}{suffix}')
#         destination_file = os.path.join(dst_dir, f'30EP_{base_id}{suffix}')
#
#         if os.path.exists(source_file):
#             shutil.copyfile(source_file, destination_file)
#             print(f"Copied {source_file} → {destination_file}")
#         else:
#             print(f"❌ Source file not found: {source_file}")
# #
# print(f'Number of cases: {len(df)}')
# print(df.columns)
# df.drop(columns = ['diff'])
# print(df.columns)
# subset = df[~df['case_id'].str.startswith('20EP_')]
#
#
# # Step 2: Compute the difference between 'dice_30' and 'dice_5'
# subset['dice_diff'] = subset['dice_30'] - subset['dice_5']
#
# # Step 3: Select rows where the difference is greater than 0.1
# final_subset = subset[subset['dice_diff'] > 0.1]
#
# print(final_subset[['dice_5','dice_30', 'dice_diff']].head(50))
# #print(len(final_subset))
#
# # Step 2: Create modified copies of these rows
# duplicated_rows = final_subset.copy()
#
# # Modify case_id by prefixing with 'EP30_'
# duplicated_rows['case_id'] = '30EP_' + duplicated_rows['case_id'].astype(str)
#
# # Set dice_5 to be equal to dice_30
# duplicated_rows['dice_5'] = duplicated_rows['dice_30']
#
# # Step 3: Append to the original df
# df = pd.concat([df, duplicated_rows], ignore_index=True)
#
# df_20 = df[df['case_id'].str.startswith('20EP_')].copy()
# df_30 = df[df['case_id'].str.startswith('30EP_')].copy()
#
# # Step 2: Extract the core ID (remove the prefixes)
# df_20['core_id'] = df_20['case_id'].str.replace('20EP_', '', regex=False)
# df_30['core_id'] = df_30['case_id'].str.replace('30EP_', '', regex=False)
#
# # Step 3: Inner join on core_id to find matches
# merged = pd.merge(df_20, df_30, on='core_id', suffixes=('_20', '_30'))
# print(merged.columns)
#
# #print(merged[['case_id_20', 'case_id_30', 'dice_5_20', 'dice_5_30']])
# print(len(merged))
#
# # Step 4: Calculate absolute difference in dice_5
# merged['dice_diff'] = (merged['dice_5_20'] - merged['dice_5_30']).abs()
#
# # Step 5: Keep only those with dice difference ≥ 0.1
# valid_matches = merged[merged['dice_diff'] >= 0.1]
#
# # Step 6: Identify EP30_ case_ids to keep
# ep30_to_keep = valid_matches['case_id_30']
#
# # Step 7: Drop all other EP30_ rows from df
# df = df[~((df['case_id'].str.startswith('EP30_')) & (~df['case_id'].isin(ep30_to_keep)))]
#
# print(len(df))
# df.to_csv('/home/bmep/plalfken/my-scratch/nnUNet/Final_dice_dist1.csv')
# # df = df.drop(columns=['tumor_class_y'])
# #
# # zero_preds = df[df['dice_5'] == 0]
# # zero_counts = zero_preds['tumor_class_x'].value_counts()
# # print("Number of cases with Dice=0 per tumor class:")
# # print(zero_counts)
# #
# #
# # zero_preds = df_adjusted[df_adjusted['dice_5'] == 0]
# # zero_counts = zero_preds['tumor_class_x'].value_counts()
# # print("Number of cases with Dice=0 per tumor class:")
# # print(zero_counts)
# #
# # to_remove = ['DES_0006', 'DES_0016', 'DES_0096', 'DES_0162']
# # df_adjusted = df_adjusted[~df_adjusted['case_id'].isin(to_remove)]
# #
# # DTF_0 = df_adjusted[(df_adjusted['tumor_class_x'] == 'DTF') & (df_adjusted['dice_20'] == 0)]
# #
# #
# #
# #
# # df_adjusted.rename(columns={'tumor_class_x': 'tumor_class'}, inplace=True)
# # df_adjusted.drop(columns=['dice_20'], inplace=True)
# # print(df_adjusted.head())
# # df_adjusted.to_csv('/home/bmep/plalfken/my-scratch/nnUNet/Final_dice_dist.csv', index=False)
# #
# #
# # #
# # # plt.figure(figsize=(14, 7))
# # # ax = sns.violinplot(
# # #     data=df,
# # #     x='tumor_class_x',
# # #     y='dice_5',
# # #     inner='box',
# # #     scale='width',
# # #     palette='Set3',
# # #     cut=0  # <-- Prevent KDE from extending beyond the data range
# # # )
# # # plt.ylim(0, 1)  # Set y-axis from 0 to 1
# # # plt.title('Dice Score Distribution per Tumor Subtype (dice_5)')
# # # plt.xlabel('Tumor Subtype')
# # # plt.ylabel('Dice Score')
# # # plt.xticks(rotation=45, ha='right')
# # # plt.tight_layout()
# # # plt.grid(True)
# # # plt.show()
# # #
# # #
# # # plt.figure(figsize=(14, 7))
# # # ax = sns.violinplot(
# # #     data=df,
# # #     x='tumor_class_x',
# # #     y='dice_5',
# # #     inner='box',
# # #     scale='width',
# # #     palette='Set3',
# # #     cut=0  # <-- Prevent KDE from extending beyond the data range
# # # )
# # # plt.ylim(0, 1)  # Set y-axis from 0 to 1
# # # plt.title('Dice Score Distribution per Tumor Subtype (dice_5)')
# # # plt.xlabel('Tumor Subtype')
# # # plt.ylabel('Dice Score')
# # # plt.xticks(rotation=45, ha='right')
# # # plt.tight_layout()
# # # plt.grid(True)
# # # plt.show()
# # #
# # # plt.figure(figsize=(14, 7))
# # # ax = sns.violinplot(
# # #     data=df_adjusted,
# # #     x='tumor_class_x',
# # #     y='dice_5',
# # #     inner='box',
# # #     scale='width',
# # #     palette='Set3',
# # #     cut=0  # <-- Prevent KDE from extending beyond the data range
# # # )
# # # plt.ylim(0, 1)  # Set y-axis from 0 to 1
# # # plt.title('Dice Score Distribution per Tumor Subtype (dice_5)')
# # # plt.xlabel('Tumor Subtype')
# # # plt.ylabel('Dice Score')
# # # plt.xticks(rotation=45, ha='right')
# # # plt.tight_layout()
# # # plt.grid(True)
# # # plt.show()
# # #
# # #
# # #
# # #
# #
# #
#
# plt.figure(figsize=(10, 6))
# sns.kdeplot(df['dice_5'], label='Dice 5', fill=True, alpha=0.5, color='skyblue')
#
# plt.title('Dice distribution of combined model results')
# plt.xlabel('Dice Score')
# plt.ylabel('Density')
# plt.legend()
# plt.grid(True)
# plt.show()
# #
plt.figure(figsize=(10, 6))
sns.kdeplot(df5['dice_5'], label='Dice 5', fill=True, alpha=0.5, color='skyblue')
sns.kdeplot(df20['dice_5'], label='Dice 20', fill=True, alpha=0.5, color='salmon')
sns.kdeplot(df30['dice_30'], label='Dice 30', fill=True, alpha=0.5, color='red')

plt.title('Overlayed Dice Score Distributions per model (5, 20, 30 epochs)')
plt.xlabel('Dice Score')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# Your plot
sns.kdeplot(df_final['dice_5'], fill=False, alpha=0.5, color='black')

# Add vertical lines for bin edges
bin_edges = [0.1, 0.5, 0.7]
for edge in bin_edges:
    plt.axvline(edge, color='black', linestyle='--', alpha=0.8)

# Add colored background spans for bins
plt.axvspan(0.0, 0.1, color='red', alpha=0.2, label='Fail (0–0.1)')
plt.axvspan(0.1, 0.5, color='salmon', alpha=0.2, label='Poor (0.1–0.5)')
plt.axvspan(0.5, 0.7, color='orange', alpha=0.2, label='Moderate (0.5–0.7)')
plt.axvspan(0.7, 1.0, color='yellowgreen', alpha=0.2, label='Good (>0.7)')

# Labels and legend
plt.xlabel('Dice Score')
plt.ylabel('Density')
plt.title('Merged Dice Score Distribution from all Models with Bins')
plt.legend()
plt.xlim(0, 1.0)  # Optional: constrain x-axis

plt.tight_layout()
plt.show()


# print(df['tumor_class_x'].value_counts())
#
#
# print(df_adjusted['tumor_class_x'].value_counts())
#
#
#
# plt.figure(figsize=(12, 6))
#
# tumor_classes = df['tumor_class_x'].unique()
#
# for tumor in tumor_classes:
#     sns.kdeplot(
#         data=df[df['tumor_class_x'] == tumor],
#         x='dice_5',
#         label=tumor,
#         fill=False,  # or True if you want area under curve
#         common_norm=False,
#         bw_adjust=1.0,  # adjust smoothing; try 0.8–1.5 if too bumpy/smooth
#         clip=(0, 1)  # ensures the KDE doesn't go beyond Dice range
#     )
#
# plt.title('Dice Score Distribution per Tumor Subtype (dice_5)')
# plt.xlabel('Dice Score')
# plt.ylabel('Density')
# plt.xlim(0, 1)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.grid(True)
# plt.show()
#
# for tumor in tumor_classes:
#     sns.kdeplot(
#         data=df_adjusted[df_adjusted['tumor_class_x'] == tumor],
#         x='dice_5',
#         label=tumor,
#         fill=False,  # or True if you want area under curve
#         common_norm=False,
#         bw_adjust=1.0,  # adjust smoothing; try 0.8–1.5 if too bumpy/smooth
#         clip=(0, 1)  # ensures the KDE doesn't go beyond Dice range
#     )
#
# plt.title('Dice Score Distribution per Tumor Subtype (dice_5)')
# plt.xlabel('Dice Score')
# plt.ylabel('Density')
# plt.xlim(0, 1)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.grid(True)
# plt.show()
#
#
# plt.figure(figsize=(10, 6))
#
# # Convert Series to NumPy array to avoid wide-form handling
# x1 = df['dice_5'].dropna().values
# x2 = df['dice_20'].dropna().values
# x3 = df_adjusted['dice_5'].dropna().values
#
# # KDEs scaled to counts using uniform weights
# sns.kdeplot(x=x1, label='Dice 5', fill=True, alpha=0.4, color='skyblue',
#             bw_adjust=0.5, common_norm=False, weights=np.ones_like(x1))
# sns.kdeplot(x=x2, label='Dice 20', fill=True, alpha=0.4, color='salmon',
#             bw_adjust=0.5, common_norm=False, weights=np.ones_like(x2))
# sns.kdeplot(x=x3, label='Adjusted Dice', fill=True, alpha=0.4, color='red',
#             bw_adjust=0.5, common_norm=False, weights=np.ones_like(x3))
#
# plt.title('Smoothed Dice Score Distributions (Scaled by Count)')
# plt.xlabel('Dice Score')
# plt.ylabel('Estimated Count')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()



# cases_cut = df[df['dice_5']==0]
# cases_cut = cases_cut[cases_cut['dice_20']!=0]
#
# cases_to_cut = ['DES_0085','DES_0077', 'DES_0116']
# cut_rows = df[df['case_id'].isin(cases_to_cut)].copy()
#
#
# cut_rows['case_id'] = '20EP_' + cut_rows['case_id']
# cut_rows['dice_5'] = cut_rows['dice_20']
# cut_rows['diff'] = 0
#
# # Step 3: Drop original rows
# df = df[~df['case_id'].isin(cases_to_cut)]
#
# # Step 4: Append modified rows
# df = pd.concat([df, cut_rows], ignore_index=True)
#
# # (Optional) Sort by case_id or reset index
# df = df.sort_values('case_id').reset_index(drop=True)
#
#
#
# # Rows where dice scores are approximately equal (within a small tolerance)
# same = df[np.isclose(df['dice_5'], df['dice_20'], atol=1e-6)]
# print(f"cases with same dice scores (within tolerance): {len(same)}")
#
# # Rows where they differ
# unique = df[~np.isclose(df['dice_5'], df['dice_20'], atol=1e-6)]
# unique['diff'] = unique['dice_20'] - unique['dice_5']
# print(f'cases with different dice scores: {len(unique)}')
#
# # Large differences
# unique_diff = unique[unique['diff'] > 0.1]
# print(f'cases with different dice scores (diff > 1): {len(unique_diff)}')
# #print(unique_diff.head(50))
#
# # Step 1: Filter rows where diff > 0.1
# to_add = unique[unique['diff'] > 0.1].copy()
#
# # Step 2: Create new case IDs
# to_add['case_id'] = '20EP_' + to_add['case_id']
#
# # Step 3: Set both dice columns to dice_20 and diff to 0
# to_add['dice_5'] = to_add['dice_20']
# to_add['diff'] = 0
#
# # Step 4: Append to the original dataframe
# df_augmented = pd.concat([df, to_add], ignore_index=True)
#
# # Optional: sort or reset index if needed
# df_augmented = df_augmented.sort_values(by='case_id').reset_index(drop=True)
# print(f'df with added cases: {len(df_augmented)}')
# df_augmented.to_csv("/scratch/bmep/plalfken/Dice_scores_adjusted.csv")