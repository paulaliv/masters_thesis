
# tabular_data_all = pd.read_csv('/home/bmep/plalfken/my-scratch/Downloads/WORC_data/WORC_all_with_nnunet_ids.csv')
# test_data = pd.read_csv('/home/bmep/plalfken/my-scratch/Downloads/WORC_data/WORC_test.csv')
# train_data = pd.read_csv('/home/bmep/plalfken/my-scratch/Downloads/WORC_data/WORC_train.csv')
# test_preprocessed = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_preprocessed/Dataset002_SoftTissue/nnUNetPlans_3d_fullres'
# # print(len(train_data))
# # print(len(test_data))
# print(len(tabular_data_all))
#
# print(test_data.columns)
# print('TEST DATA DISTRIBUTIOn')
# print(test_data['Final_Classification'].value_counts())
#
# print('TRAIN DATA DISTRIBUTION')
# print(train_data['Final_Classification'].value_counts())
#
# from sklearn.model_selection import StratifiedKFold
# X = train_data.drop(columns=['Final_Classification'])
# y = train_data['Final_Classification']
# skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
# for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
#     train_fold = y.iloc[train_idx]
#     test_fold = y.iloc[test_idx]
#
#     train_counts = train_fold.value_counts().sort_index()
#     test_counts = test_fold.value_counts().sort_index()
#
#
#
#     print(f'Fold {fold +1}')
#     print('Train Counts')
#     print(train_counts)
#     print('Test Counts')
#     print(test_counts)
import pandas as pd

import os
import numpy as np
import pandas as pd
import nibabel as nib  # or use SimpleITK if you prefer

# Paths
mask_folder = "/path/to/tumor_masks"
csv_file = "/path/to/case_metadata.csv"

# Load metadata CSV (case_id, tumor_class)
df = pd.read_csv(csv_file)

# Assume all masks have the same spacing (z,y,x) in mm
# If spacing differs per case, you need to load spacing per mask file
voxel_spacing = np.array([3.0, 1.0, 1.0])  # example: 1mm x 1mm x 1mm voxel size

volumes = []
for idx, row in df.iterrows():
    case_id = row['case_id']
    tumor_class = row['tumor_class']

    mask_path = os.path.join(mask_folder, f"{case_id}_mask.nii.gz")  # adjust naming if needed
    if not os.path.exists(mask_path):
        print(f"Mask not found for {case_id}, skipping")
        continue

    # Load mask
    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata()

    # If spacing per mask differs, get it here:
    # voxel_spacing = mask_img.header.get_zooms()[:3]

    # Count tumor voxels
    tumor_voxels = np.sum(mask > 0)

    # Compute tumor volume in mm³
    voxel_volume = np.prod(voxel_spacing)
    tumor_volume_mm3 = tumor_voxels * voxel_volume

    volumes.append({'case_id': case_id, 'tumor_class': tumor_class, 'volume_mm3': tumor_volume_mm3})

# Convert to DataFrame
vol_df = pd.DataFrame(volumes)

# Compute global stats
global_stats = vol_df['volume_mm3'].agg(['min', 'max', 'mean', 'std'])

# Compute per-subtype stats (excluding unknown if you want)
per_subtype_stats = vol_df[vol_df['tumor_class'] != 'Unknown'].groupby('tumor_class')['volume_mm3'].agg(
    ['min', 'max', 'mean', 'std'])

print("Global tumor volume stats (mm³):")
print(global_stats.round(4))

print("\nPer-subtype tumor volume stats (mm³):")
print(per_subtype_stats.round(4))
