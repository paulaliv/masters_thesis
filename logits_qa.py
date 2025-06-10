"""
The Logits-based Segmentation Quality Assessment Baseline

Logits are extracted from right before Softmax is applied in the nnunetv2 to get the final prediction.

Logits from nnUNet output should have shape [C,D,H,W]
"""
import numpy as np
import nibabel as nib
import os
import pandas as pd

preds_dir = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/Logits'
image_dir = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnUNet_raw/Dataset002_SoftTissue/labelsTs'
tabular_data_dir = r'/home/bmep/plalfken/my-scratch/Downloads/WORC_data/WORC_all_with_nnunet_ids.csv'
tabular_data = pd.read_csv(tabular_data_dir)
#print(tabular_data.head())
subtype = tabular_data[['nnunet_id','Final_Classification']]
print(subtype.head())

def compute_msp(logits, axis = 0):
    """
    Compute Maximum Softmax Probability (MSP)
    High MSP -> High confidence
    """
    e_x = np.exp(logits - np.max(logits, axis = axis, keepdims = True))
    probs = e_x / np.sum(e_x, axis = axis, keepdims = True)
    return probs.max(axis =axis), probs

def compute_entropy(probs, axis = 0):
    """Compute voxel-wise entropy from softmax probabilities."""
    return -np.sum(probs * np.log(probs + 1e-8), axis=axis)

def compute_dice(pred, gt):
    pred = np.asarray(pred == 1, dtype =np.uint8)
    gt= (np.asarray(gt == 1, dtype=np.uint8))

    intersection = np.sum(pred* gt) #true positives
    union = np.sum(pred) + np.sum(gt) #predicted plus ground truth voxels

    dice = 2. * intersection / union if union > 0 else np.nan

    return dice

def apply_temperature_scaling(logits, temperature):
    """
    Apply temperature scaling to logits before softmax
    """
    return logits / temperature

def compute_logit_gap(logits):
    """
       Compute the difference between top-1 and top-2 logits at each voxel.
       Returns: mean and std of logit gaps.
       """
    sorted_logits = np.sort(logits, axis=0)  # sort over class axis (C)
    top1 = sorted_logits[-1]  # max logit
    top2 = sorted_logits[-2]  # second-highest logit
    gap = top1 - top2
    return np.mean(gap), np.std(gap)

def collect_features():
    df = pd.DataFrame(columns=["case_id", "subtype", "dice_score","msp","mean_msp", "mean_entropy", "frac_high_entropy",
        "logit_gap_mean", "logit_gap_std", "temp_scaled_mean_msp"])

    for file in os.listdir(preds_dir):
        if file.endswith('.nii.gz'):

            stem = file.replace('.nii.gz', '')
            print(f'Processing {stem}')
            mask_dir = os.path.join(preds_dir, file)
            gt_dir = os.path.join(image_dir, stem + '.nii.gz')
            logits_dir = os.path.join(preds_dir, f'{stem}_resampled_logits.npy')
            #print(f'Ground truth directory {gt_dir}')
            #print(f'Mask directory: {mask_dir}')
            #print(f'Logits directory: {logits_dir}')
            patient = subtype[subtype['nnunet_id'] == stem]
            if not patient.empty:
                subtype_label = patient['Final_Classification'].values[0]
            else:
                subtype_label = "Unknown"
            print(f'Subtype {subtype_label}')

            logits = np.load(logits_dir)
            mask = nib.load(mask_dir).get_fdata()
            gt = nib.load(gt_dir).get_fdata()
            mask = mask.astype(np.uint8)
            #mask = np.transpose(mask, (2, 1, 0))  # from [X,Y,Z] to [Z,Y,X]
            print(f'Mask shape: {mask.shape}, GT shape: {gt.shape}, Logits shape: {logits.shape}')
            dice = compute_dice(mask, gt)
            print(f'Dice: {dice}')
            dice_scores.loc[len(dice_scores)] = [stem, subtype_label, dice]
    print(f"Processed {len(dice_scores)} cases")
    return dice_scores


dice_scores = collect_features()
print(dice_scores.head(13))



