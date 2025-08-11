import os
import numpy
import sys
from typing import Tuple, Optional
import nibabel as nib
import numpy as np
import json
import SimpleITK as sitk
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.ndimage import label, find_objects
import glob
from scipy.ndimage import zoom
import pandas as pd


def bin_dice_score( dice):
    epsilon = 1e-8
    dice = np.asarray(dice)
    dice_adjusted = dice - epsilon  # Shift slightly left
    bin_edges = [0.1, 0.5, 0.7]
    return np.digitize(dice_adjusted, bin_edges, right=True)


def preprocess_folder(data, splits):
    # subtypes_csv = "/gpfs/home6/palfken/masters_thesis/all_folds"

    clinical_data = "/gpfs/home6/palfken/masters_thesis/Final_dice_dist1.csv"
    dice_df = pd.read_csv(clinical_data)
    dice_df.set_index('case_id', inplace=True)  # for quick lookup

    train_case_ids = splits[0]["train"]
    val_case_ids = splits[0]["val"]
    case_ids = train_case_ids + val_case_ids

    #
    # image_paths = []
    # for idir in image_dirs:
    #     image_paths.extend(glob.glob(os.path.join(idir, '*_0000.nii.gz')))
    # image_paths = sorted(image_paths)

    all_stats = {}

    umap_types = ["epkl", "confidence", "entropy", "mutual_info"]  # Your uncertainty maps

    for case_id in case_ids:
        stats_dict = {}

        # Load dice score from CSV if present, else NaN
        dice_score = dice_df.loc[case_id, 'dice_5'] if case_id in dice_df.index else np.nan
        stats_dict['dice'] = dice_score

        # Load predicted mask and ground truth mask
        pred_path = os.path.join(data, f"{case_id}_pred.npy")
        mask_path = os.path.join(data, f"{case_id}_mask.npy")

        if not (os.path.exists(pred_path) and os.path.exists(mask_path)):
            print(f"Missing mask or pred for case {case_id}, skipping.")
            continue

        resampled_pred = np.load(pred_path)
        resampled_mask = np.load(mask_path)

        # For each uncertainty map, load and compute statistics
        for umap_type in umap_types:
            umap_path = os.path.join(data, f"{case_id}_{umap_type}.npy")
            if not os.path.exists(umap_path):
                print(f"Missing uncertainty map {umap_type} for case {case_id}, skipping this map.")
                continue

            resampled_umap = np.load(umap_path)

            gt_mask = (resampled_mask > 0)
            pred_mask = (resampled_pred > 0)

            gt_mean_unc = np.mean(resampled_umap[gt_mask]) if gt_mask.sum() > 0 else np.nan
            pred_mean_unc = np.mean(resampled_umap[pred_mask]) if pred_mask.sum() > 0 else np.nan
            full_mean_unc = np.mean(resampled_umap)

            gt_median_unc = np.median(resampled_umap[gt_mask]) if gt_mask.sum() > 0 else np.nan
            pred_median_unc = np.median(resampled_umap[pred_mask]) if pred_mask.sum() > 0 else np.nan

            gt_std_unc = np.std(resampled_umap[gt_mask]) if gt_mask.sum() > 0 else np.nan
            pred_std_unc = np.std(resampled_umap[pred_mask]) if pred_mask.sum() > 0 else np.nan

            ratio_pred_gt_unc = pred_mean_unc / gt_mean_unc if gt_mean_unc and not np.isnan(gt_mean_unc) else np.nan

            # Store with prefixed keys
            stats_dict[f"{umap_type}_gt_mean_unc"] = gt_mean_unc
            stats_dict[f"{umap_type}_pred_mean_unc"] = pred_mean_unc
            stats_dict[f"{umap_type}_full_mean_unc"] = full_mean_unc
            stats_dict[f"{umap_type}_gt_median_unc"] = gt_median_unc
            stats_dict[f"{umap_type}_pred_median_unc"] = pred_median_unc
            stats_dict[f"{umap_type}_gt_std_unc"] = gt_std_unc
            stats_dict[f"{umap_type}_pred_std_unc"] = pred_std_unc
            stats_dict[f"{umap_type}_ratio_pred_gt_unc"] = ratio_pred_gt_unc

        all_stats[case_id] = stats_dict



    df = pd.DataFrame.from_dict(all_stats, orient='index')

    df['dice_bin'] = bin_dice_score(df['dice'].values)

    print("\nDice Score Distribution and Uncertainty Stats by Tumor Class:")

    # Find all uncertainty-related columns
    unc_cols = [c for c in df.columns if c.endswith('unc')]


    print("\nUncertainty stats per Dice bin:")
    for bin_id, bin_group in df.groupby('dice_bin'):
        # Map bin_id to readable range
        if bin_id == 0:
            bin_range = f"<= 0.1"
        elif bin_id == 1:
            bin_range = "0.1 < dice <= 0.5"
        elif bin_id == 2:
            bin_range = "0.5 < dice <= 0.7"
        else:
            bin_range = "> 0.7"

        print(f" Dice bin {bin_id} ({bin_range}): {len(bin_group)} samples")
        if len(bin_group) > 0 and unc_cols:
            for col in unc_cols:
                col_mean = bin_group[col].mean()
                col_std = bin_group[col].std()
                print(f"  {col}: {col_mean:.3f} ± {col_std:.3f}")
        else:
            print("  No samples in this bin.")




def main():

    data = "/gpfs/home6/palfken/QA_dataTr_final/"
    with open('/gpfs/home6/palfken/masters_thesis/Final_splits30.json', 'r') as f:
        splits = json.load(f)



    preprocess_folder(data, splits)

if __name__ == '__main__':
    main()

