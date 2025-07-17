from radiomics import featureextractor
import logging
import os
import pandas as pd
import nibabel as nib
import numpy as np
import SimpleITK as sitk

settings = {
    'geometryTolerance': 2  # try small values like 1 or 2 (units = mm)
}

extractor = featureextractor.RadiomicsFeatureExtractor(**settings)


# === Paths ===
data_dir = '/gpfs/home6/palfken/QA_dataTr_final'
output_csv = '/gpfs/home6/palfken/radiomics_features.csv'
subtypes_csv = "/gpfs/home6/palfken/masters_thesis/all_folds.csv"
subtypes_df = pd.read_csv(subtypes_csv)
print(f'Number of cases in csv file: {subtypes_df.shape[0]}')
# === Collect results ===
all_features = []

for filename in os.listdir(data_dir):
    if not filename.endswith('_img.npy'):
        continue

    # Find corresponding mask
    base_id = filename.replace('_img.npy', '')
    print(f'Processing {base_id}')
    image_path = os.path.join(data_dir, f'{base_id}_img.npy')
    mask_path = os.path.join(data_dir, f'{base_id}_mask.npy')


    # Read image and mask
    image = np.load(image_path)
    mask = np.load(mask_path)

    if not os.path.exists(mask_path):
        print(f'Skipping {base_id}: mask not found')
        continue

    try:
        features = extractor.execute(image, mask)
        features['case_id'] = base_id

        # Add subtype if it exists
        subtype_row = subtypes_df[subtypes_df['case_id'] == base_id]
        if not subtype_row.empty:
            features['subtype'] = subtype_row.iloc[0]['subtype']
        else:
            features['subtype'] = 'Unknown'
            print(f'Case id {base_id}: no subtype in csv file!')

        all_features.append(features)
    except Exception as e:
        print(f"Failed on {base_id}: {e}")
        continue

# === Convert to DataFrame and save ===
df = pd.DataFrame(all_features)
print(f'Number of cases in final dataframe: {df.shape[0]}')
df.set_index('case_id', inplace=True)
df.to_csv(output_csv)
print(f"Saved features to: {output_csv}")