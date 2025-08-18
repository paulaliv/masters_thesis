import os

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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score, average_precision_score

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
        stats_dict['case_id'] = case_id

        # Load predicted mask and ground truth mask
        pred_path = os.path.join(data, f"{case_id}_pred.npy")


        if not (os.path.exists(pred_path)):
            print(f"Missing mask or pred for case {case_id}, skipping.")
            return None

        resampled_pred = np.load(pred_path)


        # For each uncertainty map, load and compute statistics
        for umap_type in umap_types:
            umap_path = os.path.join(data, f"{case_id}_{umap_type}.npy")
            if not os.path.exists(umap_path):
                print(f"Missing uncertainty map {umap_type} for case {case_id}, skipping this map.")
                continue

            resampled_umap = np.load(umap_path)

            pred_mask = (resampled_pred > 0)

            pred_mean_unc = np.mean(resampled_umap[pred_mask]) if pred_mask.sum() > 0 else np.nan
            full_mean_unc = np.mean(resampled_umap)


            pred_median_unc = np.median(resampled_umap[pred_mask]) if pred_mask.sum() > 0 else np.nan

            pred_std_unc = np.std(resampled_umap[pred_mask]) if pred_mask.sum() > 0 else np.nan

            # Store with prefixed keys
            stats_dict[f"{umap_type}_pred_mean_unc"] = pred_mean_unc
            stats_dict[f"{umap_type}_full_mean_unc"] = full_mean_unc
            stats_dict[f"{umap_type}_pred_median_unc"] = pred_median_unc
            stats_dict[f"{umap_type}_pred_std_unc"] = pred_std_unc

        all_stats[case_id] = stats_dict
    return all_stats


def preprocess_folder_1(data):
    # subtypes_csv = "/gpfs/home6/palfken/masters_thesis/all_folds"
    ood_dir = "/home/bmep/plalfken/my-scratch/test_table1.csv"

    #clinical_data = "/gpfs/home6/palfken/masters_thesis/Final_dice_dist1.csv"
    df = pd.read_csv(ood_dir)
    case= 'STT_0486'

    df_unique = df.drop_duplicates(subset='new_accnr')
    df_unique = df_unique[df_unique['new_accnr'] != case]


    #dice_df.set_index('case_id', inplace=True)  # for quick lookup

    case_ids = df_unique["new_accnr"].values
    subtype = df_unique["tumor_class"].values
    dist = df_unique["dist"].values
    # image_paths = []
    # for idir in image_dirs:
    #     image_paths.extend(glob.glob(os.path.join(idir, '*_0000.nii.gz')))
    # image_paths = sorted(image_paths)

    all_stats = {}

    umap_types = ["epkl", "confidence", "entropy", "mutual_info"]  # Your uncertainty maps

    for idx,case_id in enumerate(case_ids):
        stats_dict = {}

        stats_dict['case_id']= case_id
        #
        stats_dict['subtype'] = subtype[idx]  # add tumor subtype
        stats_dict['dist'] = dist[idx]

        pred_path = os.path.join(data, f"{case_id}_pred.npy")


        if not (os.path.exists(pred_path)):
            print(f"Missing mask or pred for case {case_id}, skipping.")
            continue

        resampled_pred = np.load(pred_path)


        # For each uncertainty map, load and compute statistics
        for umap_type in umap_types:
            umap_path = os.path.join(data, f"{case_id}_{umap_type}.npy")
            if not os.path.exists(umap_path):
                print(f"Missing uncertainty map {umap_type} for case {case_id}, skipping this map.")
                continue

            resampled_umap = np.load(umap_path)

            pred_mask = (resampled_pred > 0)


            pred_mean_unc = np.mean(resampled_umap[pred_mask]) if pred_mask.sum() > 0 else np.nan
            full_mean_unc = np.mean(resampled_umap)


            pred_median_unc = np.median(resampled_umap[pred_mask]) if pred_mask.sum() > 0 else np.nan


            pred_std_unc = np.std(resampled_umap[pred_mask]) if pred_mask.sum() > 0 else np.nan

            # Store with prefixed keys
            stats_dict[f"{umap_type}_pred_mean_unc"] = pred_mean_unc
            stats_dict[f"{umap_type}_full_mean_unc"] = full_mean_unc
            stats_dict[f"{umap_type}_pred_median_unc"] = pred_median_unc

            stats_dict[f"{umap_type}_pred_std_unc"] = pred_std_unc


        all_stats[case_id] = stats_dict

    return all_stats



def main():

    data = "/home/bmep/plalfken/my-scratch/test_unc_maps_BAD/"
    umc_id_dir = "/home/bmep/plalfken/my-scratch/unc_features_id.csv"
    #umc_ood_dir = "/home/bmep/plalfken/my-scratch/unc_features_id.csv"
    worc_id_df = pd.read_csv(umc_id_dir)
    umc_preds = {
        'mutual_info': "/home/bmep/plalfken/my-scratch/results/mutual_info_ood_results.csv",
        'epkl': "/home/bmep/plalfken/my-scratch/results/mutual_info_ood_results.csv",
        'entropy': "/home/bmep/plalfken/my-scratch/results/mutual_info_ood_results.csv",
        'confidence': "/home/bmep/plalfken/my-scratch/results/mutual_info_ood_results.csv"
    }
    worc_preds = {
        'mutual_info': "/home/bmep/plalfken/my-scratch/results/mutual_info_id_results_mask.csv",
        'epkl': "/home/bmep/plalfken/my-scratch/results/mutual_info_id_results_mask.csv",
        'entropy': "/home/bmep/plalfken/my-scratch/results/mutual_info_id_results_mask.csv",
        'confidence': "/home/bmep/plalfken/my-scratch/results/mutual_info_id_results_mask.csv"
    }

    # data1 = "/gpfs/home6/palfken/ood_features/ood_umaps_cropped_30/"
    # with open('/gpfs/home6/palfken/masters_thesis/Final_splits30.json', 'r') as f:
    #     splits = json.load(f)
    ood_stats = preprocess_folder_1(data)



    df_ood = pd.DataFrame.from_dict(ood_stats, orient='index')
    print(f'{len(df_ood)} Cases in df_ood')
    print(f'df_ood columns: {df_ood.columns}')

    # --- Add WORC predictions ---
    for metric, path in worc_preds.items():
        pred_df = pd.read_csv(path)  # should contain columns ['case_id', 'preds']
        print(f'pred_df columns: {pred_df.columns}')
        pred_df = pred_df.drop_duplicates(subset=['case_id'])

        # Rename the prediction column to include the metric
        pred_df = pred_df.rename(columns={'preds': f'pred_{metric}'})

        # Merge with WORC ID dataframe on case_id
        worc_id_df = worc_id_df.merge(pred_df[['case_id', f'pred_{metric}']], on='case_id', how='left')

    # --- Add UMC predictions ---
    for metric, path in umc_preds.items():
        pred_df = pd.read_csv(path)  # should contain columns ['case_id', 'maj_preds']
        pred_df = pred_df.drop_duplicates(subset=['case_id'])

        # Rename the prediction column to include the metric
        pred_df = pred_df.rename(columns={'maj_pred': f'pred_{metric}'})

        # Merge with OOD dataframe on new_accnr == case_id
        df_ood = df_ood.merge(pred_df[['case_id', f'pred_{metric}']],on='case_id', how='left')



    print(f'OOD FOLDER: {len(df_ood)}')
    print(df_ood.head(10))


    df_ood.to_csv('/home/bmep/plalfken/my-scratch/unc_features_test.csv', index=False)

    umc_id = df_ood[df_ood['dist'] == 'ID']
    print(f'Number of WORD ID Samples: {len(umc_id)}')
    umc_ood = df_ood[df_ood['dist'] == 'OOD']
    print(f'Number of WORD OOD Samples: {len(umc_ood)}')


    # Metrics & plotting
    metrics = ['mutual_info', 'epkl', 'entropy', 'confidence']
    dice_bins = sorted(worc_id_df['dice_bin'].unique())
    colors = {'WORC ID': 'blue', 'UMC OOD': 'red', 'UMC ID': 'green'}

    for metric in metrics:
        print(f"Processing metric: {metric}")
        umc_id_clean = umc_id.dropna(subset=[f'{metric}_pred_mean_unc', f'pred_{metric}'])
        unc_id = umc_id_clean[f'{metric}_pred_mean_unc'].values
        pred_id = umc_id_clean[f'pred_{metric}'].values

        # Similarly for WORC
        worc_id_clean = worc_id_df.dropna(subset=[f'{metric}_pred_mean_unc', f'pred_{metric}'])
        unc_worc = worc_id_clean[f'{metric}_pred_mean_unc'].values
        pred_worc = worc_id_clean[f'pred_{metric}'].values

        # And for UMC OOD
        umc_ood_clean = umc_ood.dropna(subset=[f'{metric}_pred_mean_unc', f'pred_{metric}'])
        unc_ood = umc_ood_clean[f'{metric}_pred_mean_unc'].values
        pred_ood = umc_ood_clean[f'pred_{metric}'].values

        # KDE log-likelihood
        kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(unc_worc[:, None])
        log_prob_id = kde.score_samples(unc_id[:, None])
        log_prob_ood = kde.score_samples(unc_ood[:, None])

        plt.hist(log_prob_id, bins=30, alpha=0.6, label='ID')
        plt.hist(log_prob_ood, bins=30, alpha=0.6, label='OOD')
        plt.xlabel('Log-Likelihood under ID KDE')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(f"/home/bmep/plalfken/my-scratch/log_prob_{metric}.png")
        plt.show()
        plt.close()

        # AUROC / AUPR
        for label, scores in zip(['ID', 'OOD'], [unc_id, unc_ood]):
            y_true = np.concatenate([np.zeros(len(unc_worc)), np.ones(len(scores))])
            y_scores = np.concatenate([unc_worc, scores])
            roc = roc_auc_score(y_true, y_scores)
            pr = average_precision_score(y_true, y_scores)
            print(f"AUROC for WORC vs {label} ({metric}): {roc:.3f}, AUPR: {pr:.3f}")

        # Violin / KDE plots
        df_plot = pd.concat([
            pd.DataFrame({'uncertainty': unc_worc, 'dist': 'WORC ID', 'dice_bin': pred_worc}),
            pd.DataFrame({'uncertainty': unc_id, 'dist': 'UMC ID', 'dice_bin': pred_id}),
            pd.DataFrame({'uncertainty': unc_ood, 'dist': 'UMC OOD', 'dice_bin': pred_ood})
        ])

        plt.figure(figsize=(12, 6))
        sns.countplot(
            data=df_plot,
            x='dice_bin',  # your already-binned Dice predictions
            hue='dist',  # the three datasets
            palette=colors
        )
        plt.xlabel('Dice Bin Predictions')
        plt.ylabel('Frequency')
        plt.title(f'Frequency of Dice Bin Predictions per Dataset ({metric})')
        plt.legend(title='Dataset')
        plt.tight_layout()
        plt.savefig(f"/home/bmep/plalfken/my-scratch/dicebin_frequency_{metric}.png")
        plt.show()


        plt.figure(figsize=(10, 6))
        sns.violinplot(x='dice_bin', y='uncertainty', hue='dist', data=df_plot,
                       inner='quartile', palette=colors)
        plt.title(f'Predicted Mean {metric} by Dice Bin')
        plt.tight_layout()
        plt.savefig(f"/home/bmep/plalfken/my-scratch/violin_{metric}_all_three.png")
        plt.show()
        plt.close()

        plt.figure(figsize=(12, 6))
        for dist_name, color in colors.items():
            sns.kdeplot(df_plot[df_plot['dist'] == dist_name]['uncertainty'],
                        fill=True, alpha=0.4, color=color, label=dist_name)
            plt.axvline(df_plot[df_plot['dist'] == dist_name]['uncertainty'].mean(),
                        color=color, linestyle='--')
        plt.title(f' Global Uncertainty Distribution : {metric}')
        plt.xlabel('Uncertainty')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"/home/bmep/plalfken/my-scratch/kde_{metric}_all_three.png")
        plt.show()
        plt.close()
    # print("\nDice Score Distribution and Uncertainty Stats by Tumor Class:")
    #
    # # Select columns for predicted region mean uncertainty only (can adjust if you want gt or full)
    # unc_cols_pred_worc = [c for c in worc_id_df.columns if c.endswith('pred_mean_unc')]
    # unc_cols_pred_id = [c for c in umc_id.columns if c.endswith('pred_mean_unc')]
    # unc_cols_pred_ood = [c for c in umc_ood.columns if c.endswith('pred_mean_unc')]
    #
    # dice_bins = sorted(worc_id_df['dice_bin'].unique())
    # colors = {'WORC': 'blue', 'UMC OOD': 'red', 'UMC ID': 'green'}
    # # For simplicity, if multiple pred_mean_unc columns exist, take the first
    # metrics = ['EPKL','Confidence','Entropy','Mutual-Info']
    # for idx, metric in enumerate(unc_cols_pred_worc):
    #     print(f'METRICS FOR {metric}')
    #     unc_col = unc_cols_pred_worc[idx]
    #     unc_col_id = unc_cols_pred_id[idx]
    #     unc_col_ood = unc_cols_pred_ood[idx]
    #
    #
    #     # Extract and clean ID values
    #     worc_unc = worc_id_df[unc_col].values
    #     print("NaNs in ID:", np.isnan(worc_unc).sum())
    #     worc_unc_clean = worc_unc[~np.isnan(worc_unc)]
    #
    #     # Extract and clean OOD values
    #     id_unc = umc_id[unc_col_id].values
    #     print("NaNs in OOD before cleaning:", np.isnan(id_unc).sum())
    #     id_unc_clean = id_unc[~np.isnan(id_unc)]
    #     print("NaNs in OOD after cleaning:", np.isnan(id_unc_clean).sum())
    #
    #     # Extract and clean OOD values
    #     ood_unc = umc_ood[unc_col_ood].values
    #     print("NaNs in OOD before cleaning:", np.isnan(ood_unc).sum())
    #     ood_unc_clean = ood_unc[~np.isnan(ood_unc)]
    #     print("NaNs in OOD after cleaning:", np.isnan(ood_unc_clean).sum())
    #
    #     kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(worc_unc_clean[:, None])
    #     log_prob = kde.score_samples(ood_unc_clean[:, None])
    #     log_prob_id = kde.score_samples(id_unc_clean[:, None])
    #
    #     plt.hist(log_prob_id, bins=30, alpha=0.6, label='ID')
    #     plt.hist(log_prob, bins=30, alpha=0.6, label='OOD')
    #     plt.xlabel('Log-Likelihood under ID KDE')
    #     plt.ylabel('Count')
    #     plt.legend()
    #     plt.savefig(f"/home/bmep/plalfken/my-scratch/log_prob_{metric}.png")
    #     plt.close()
    #
    #     from sklearn.metrics import roc_auc_score, average_precision_score
    #     y_true = np.concatenate([np.zeros(len(worc_unc_clean)), np.ones(len(id_unc_clean))])
    #     scores = np.concatenate([worc_unc_clean,id_unc_clean])
    #
    #     roc = roc_auc_score(y_true, scores)
    #     pr = average_precision_score(y_true, scores)
    #
    #     print('AUROC FOR WORC AND ID UMC DATA')
    #     print(f"AUROC: {roc:.3f}, AUPR: {pr:.3f}")
    #
    #     y_true = np.concatenate([np.zeros(len(worc_unc_clean)), np.ones(len(ood_unc_clean))])
    #     scores = np.concatenate([worc_unc_clean, ood_unc_clean])
    #
    #     roc = roc_auc_score(y_true, scores)
    #     pr = average_precision_score(y_true, scores)
    #
    #     print('AUROC FOR WORC AND OOD UMC DATA')
    #     print(f"AUROC: {roc:.3f}, AUPR: {pr:.3f}")
    #
    #     # --- Combine for plotting ---
    #
    #     # --- Combine for plotting ---
    #     df_plot = pd.concat([
    #         pd.DataFrame({'uncertainty': worc_id_df[unc_col], 'dist': 'WORC ID'}),
    #         pd.DataFrame({'uncertainty': umc_id[unc_col_id], 'dist': 'UMC ID'}),
    #         pd.DataFrame(
    #             {'uncertainty': umc_ood[unc_col_ood], 'dist': 'UMC OOD'})
    #     ])
    #     plt.figure(figsize=(10, 6))
    #     sns.violinplot(x='dice_bin', y='uncertainty', hue='dist', data=df_plot,
    #                    inner='quartile', palette={'WORC ID': 'blue', 'UMC ID': 'green', 'UMC OOD': 'red'})
    #     plt.title(f'Predicted Mean {metrics[idx]} by Dice Bin')
    #     plt.xlabel('Dice Bin')
    #     plt.ylabel('Predicted Mean Uncertainty')
    #     plt.legend(title='Distribution')
    #     plt.tight_layout()
    #     plt.savefig(f"/home/bmep/plalfken/my-scratch/violin_{metrics[idx]}_all_three.png")
    #     plt.close()
    #
    #     plt.figure(figsize=(12, 6))
    #     for dist_name, color in zip(['WORC ID', 'UMC ID', 'UMC OOD'], ['blue', 'green', 'red']):
    #         sns.kdeplot(df_plot[df_plot['dist'] == dist_name]['uncertainty'],
    #                     fill=True, alpha=0.4, color=color, label=dist_name)
    #     for dist_name, color in zip(['WORC ID', 'UMC ID', 'UMC OOD'], ['blue', 'green', 'red']):
    #         plt.axvline(df_plot[df_plot['dist'] == dist_name]['uncertainty'].mean(), color=color, linestyle='--')
    #     plt.title(f'Uncertainty Distribution: {metrics[idx]}')
    #     plt.xlabel('Predicted Mean Uncertainty')
    #     plt.ylabel('Density')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(f"/home/bmep/plalfken/my-scratch/kde_{metrics[idx]}_all_three.png")
    #     plt.close()
    #
    #
    #
    #     plt.figure(figsize=(12, 6))
    #
    #     for i, bin_label in enumerate(dice_bins):
    #         plt.subplot(1, len(dice_bins), i + 1)
    #
    #         id_unc = df.loc[df['dice_bin'] == bin_label, unc_col]
    #         ood_unc = df_ood.loc[df_ood['dice_bin'] == bin_label, unc_col_ood]
    #
    #         # Smooth KDE plots
    #         sns.kdeplot(id_unc, fill=True, alpha=0.4, color=colors['In-Distribution'], label='ID')
    #         sns.kdeplot(ood_unc, fill=True, alpha=0.4, color=colors['OOD'], label='OOD')
    #
    #         # Mean lines
    #         plt.axvline(id_unc.mean(), color=colors['In-Distribution'], linestyle='--')
    #         plt.axvline(ood_unc.mean(), color=colors['OOD'], linestyle='--')
    #
    #         plt.title(f'Dice Bin: {bin_label}')
    #         plt.xlabel(f'Predicted Mean {metrics[idx]}')
    #         plt.ylabel('Density')
    #         if i == 0:
    #             plt.legend()
    #
    #     plt.tight_layout()
    #     plt.savefig(f"/gpfs/home6/palfken/hist_kde_{metric}.png")
    #     plt.close()
    #
    # # for col in unc_cols_pred:
    # #     means = df.groupby('dice_bin')[col].mean()
    # #     stds = df.groupby('dice_bin')[col].std()
    # #
    # #     plt.errorbar(means.index, means.values, yerr=stds.values, label=col, capsize=5, marker='o')
    # #
    # # plt.xlabel('Dice bin')
    # # plt.ylabel('Mean Uncertainty ± std')
    # # plt.title('Uncertainty metrics by Dice bins')
    # # plt.legend()
    # # plt.xticks(ticks=[0, 1, 2, 3], labels=['<=0.1', '0.1-0.5', '0.5-0.7', '>0.7'])
    # # plt.savefig("/gpfs/home6/palfken/mean_uncertainty.png")
    #
    # print("\nUncertainty stats per Dice bin:ID")
    # for bin_id, bin_group in df.groupby('dice_bin'):
    #     # Map bin_id to readable range
    #     if bin_id == 0:
    #         bin_range = f"<= 0.1"
    #     elif bin_id == 1:
    #         bin_range = "0.1 < dice <= 0.5"
    #     elif bin_id == 2:
    #         bin_range = "0.5 < dice <= 0.7"
    #     else:
    #         bin_range = "> 0.7"
    #
    #     print(f" Dice bin {bin_id} ({bin_range}): {len(bin_group)} samples")
    #     if len(bin_group) > 0 and unc_cols_pred:
    #         for col in unc_cols_pred:
    #             col_mean = bin_group[col].mean()
    #             col_std = bin_group[col].std()
    #             print(f"  {col}: {col_mean:.3f} ± {col_std:.3f}")
    #     else:
    #         print("  No samples in this bin.")
    #
    # print("\nUncertainty stats per Dice bin:OOD")
    # for bin_id, bin_group in df_ood.groupby('dice_bin'):
    #     # Map bin_id to readable range
    #     if bin_id == 0:
    #         bin_range = f"<= 0.1"
    #     elif bin_id == 1:
    #         bin_range = "0.1 < dice <= 0.5"
    #     elif bin_id == 2:
    #         bin_range = "0.5 < dice <= 0.7"
    #     else:
    #         bin_range = "> 0.7"
    #
    #     print(f" Dice bin {bin_id} ({bin_range}): {len(bin_group)} samples")
    #     if len(bin_group) > 0 and unc_cols_pred_ood:
    #         for col in unc_cols_pred_ood:
    #             col_mean = bin_group[col].mean()
    #             col_std = bin_group[col].std()
    #             print(f"  {col}: {col_mean:.3f} ± {col_std:.3f}")
    #     else:
    #         print("  No samples in this bin.")
    #

if __name__ == '__main__':
    main()

