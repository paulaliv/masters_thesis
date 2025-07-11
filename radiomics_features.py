from radiomics import featureextractor
import logging
import os
import pandas as pd
import nibabel as nib

import SimpleITK as sitk

extractor = featureextractor.RadiomicsFeatureExtractor()


# === Paths ===
data_dir = "/gpfs/home6/palfken/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/COMPLETE_nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/Cropped_nifti/"
output_csv = '/gpfs/home6/palfken/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/COMPLETE_nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/radiomics_features.csv'
subtypes_csv = "/gpfs/home6/palfken/masters_thesis/all_folds.csv"
subtypes_df = pd.read_csv(subtypes_csv)
print(f'Number of cases in csv file: {subtypes_df.shape[0]}')
# === Collect results ===
all_features = []

for filename in os.listdir(data_dir):
    if not filename.endswith('resized_ROI_image.nii.gz'):
        continue

    # Find corresponding mask
    base_id = filename.replace('_resized_ROI_image.nii.gz', '')
    print(f'Processing {base_id}')
    image_path = os.path.join(data_dir, f'{base_id}_resized_ROI_image.nii.gz')
    mask_path = os.path.join(data_dir, f'{base_id}_resized_ROI_mask.nii.gz')


    # Read image and mask
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)


    # Check if image has a channel dimension and squeeze it
    if image.GetDimension() == 4 and image.GetSize()[0] == 1:
        # Remove the first dimension (channel = 1)
        size = list(image.GetSize())  # e.g., [1, 48, 272, 256]
        index = [0, 0, 0, 0]  # start from the beginning in all dims
        size[0] = 0  # we want to discard channel dim


        image = sitk.Extract(image, size=size, index=index)
        print(f'Resized image: {image.GetSize()}')

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