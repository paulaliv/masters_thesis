

import numpy as np
import os
import pandas as pd
import nibabel as nib


preds_dir = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/predictions_ood'
image_dir = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnUNet_raw/Dataset002_SoftTissue/labelsTs'

tabular_data_dir = r'/home/bmep/plalfken/my-scratch/Downloads/WORC_data/WORC_all_with_nnunet_ids.csv'
tabular_data = pd.read_csv(tabular_data_dir)
subtype = tabular_data[['nnunet_id','Final_Classification']]

def compute_dice(pred, gt):
    pred = np.asarray(pred == 1, dtype =np.uint8)
    gt= (np.asarray(gt == 1, dtype=np.uint8))

    intersection = np.sum(pred* gt) #true positives
    union = np.sum(pred) + np.sum(gt) #predicted plus ground truth voxels

    dice = 2. * intersection / union if union > 0 else np.nan

    return dice


for file in os.listdir(preds_dir):
    if file.endswith('.nii.gz'):
        stem = file.replace('.nii.gz', '')
        print(f'Processing {stem}')
        # === retrieve patient data ===
        mask_dir = os.path.join(preds_dir, file)
        gt_dir = os.path.join(image_dir, stem + '.nii.gz')

        mask = nib.load(mask_dir).get_fdata()
        gt = nib.load(gt_dir).get_fdata()
        mask = mask.astype(np.uint8)
        patient = subtype[subtype['nnunet_id'] == stem]
        if not patient.empty:
            subtype_label = patient['Final_Classification'].values[0]
        else:
            subtype_label = "Unknown"
        print(f'Subtype {subtype_label}')
        # print(f'Mask shape: {mask.shape}, GT shape: {gt.shape}, Logits shape: {logits.shape}')

        # === compute all features ===
        dice = compute_dice(mask, gt)
        print(f'Dice: {dice}')