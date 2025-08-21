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


def preprocess_folder_1(data):
    # subtypes_csv = "/gpfs/home6/palfken/masters_thesis/all_folds"
    ood_dir = "/gpfs/home6/palfken/Dice_scores_OOD_30.csv"

    #clinical_data = "/gpfs/home6/palfken/masters_thesis/Final_dice_dist1.csv"
    df = pd.read_csv(ood_dir)
    df_unique = df.drop_duplicates(subset='case_id')

    df_unique.to_csv(ood_dir, index=False)

    dice_df = df_unique[df_unique['tumor_class'] == 'Lipoma']
    #dice_df.set_index('case_id', inplace=True)  # for quick lookup

    case_ids = dice_df["case_id"].values
    dice_scores = dice_df["dice"].values
    # image_paths = []
    # for idir in image_dirs:
    #     image_paths.extend(glob.glob(os.path.join(idir, '*_0000.nii.gz')))
    # image_paths = sorted(image_paths)

    all_stats = {}

    umap_types = ["epkl", "confidence", "entropy", "mutual_info"]  # Your uncertainty maps

    for idx,case_id in enumerate(case_ids):
        stats_dict = {}

        # Load dice score from CSV if present, else NaN
        dice_score = dice_scores[idx]
        print(f"{case_id}: {dice_score}")
        stats_dict['dice'] = dice_score

        # Load predicted mask and ground truth mask
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

    data = "/gpfs/home6/palfken/QA_dataTr_final/"
    data1 = "/gpfs/home6/palfken/ood_features/ood_umaps_cropped_30/"
    with open('/gpfs/home6/palfken/masters_thesis/Final_splits30.json', 'r') as f:
        splits = json.load(f)



    # id_stats = preprocess_folder(data, splits)
    #
    # df = pd.DataFrame.from_dict(id_stats, orient='index')
    # df.to_csv("/gpfs/home6/palfken/ood_features/id_stats.csv")
    df = pd.read_csv("/gpfs/home6/palfken/ood_features/id_stats.csv")
    print(f'ID FOLDER: {len(df)}')

    df['dice_bin'] = bin_dice_score(df['dice'].values)

    # ood_stats = preprocess_folder_1(data1)
    #
    # df_ood = pd.DataFrame.from_dict(ood_stats, orient='index')
    # print(f'OOD FOLDER: {len(df_ood)}')
    # print(df_ood.head(10))
    #
    # df_ood['dice_bin'] = bin_dice_score(df_ood['dice'].values)
    # df_ood.to_csv("/gpfs/home6/palfken/ood_features/ood_stats.csv")

    df_ood = pd.read_csv("/gpfs/home6/palfken/ood_features/ood_stats.csv")
    print("\nDice Score Distribution and Uncertainty Stats by Tumor Class:")

    # Select columns for predicted region mean uncertainty only (can adjust if you want gt or full)
    unc_cols_pred = [c for c in df.columns if c.endswith('pred_mean_unc')]
    unc_cols_pred_ood = [c for c in df_ood.columns if c.endswith('pred_mean_unc')]

    dice_bins = sorted(df['dice_bin'].unique())
    colors = {'In-Distribution': 'blue', 'OOD': 'red'}
    # For simplicity, if multiple pred_mean_unc columns exist, take the first
    metrics = ['EPKL','Confidence','Entropy','Mutual-Info']

    dice_bins = sorted(df['dice_bin'].unique())
    results = []

    for idx, metric in enumerate(unc_cols_pred):
        unc_col = unc_cols_pred[idx]
        unc_col_ood = unc_cols_pred_ood[idx]
        print(f"\n=== METRIC: {metric} ===")

        for bin_id in dice_bins:
            print(f"--- Dice bin {bin_id} ---")

            # Select bin-specific ID and OOD
            id_subset = df[df['dice_bin'] == bin_id]
            ood_subset = df_ood[df_ood['dice_bin'] == bin_id]

            if len(id_subset) == 0 or len(ood_subset) == 0:
                print(f"Skipping bin {bin_id}, not enough data.")
                continue

            id_unc = id_subset[unc_col].values
            ood_unc = ood_subset[unc_col_ood].values

            # Clean NaNs
            id_unc = id_unc[~np.isnan(id_unc)]
            ood_unc = ood_unc[~np.isnan(ood_unc)]

            if len(id_unc) == 0 or len(ood_unc) == 0:
                print(f"Skipping bin {bin_id}, all NaNs.")
                continue

            # KDE fit on ID
            kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(id_unc[:, None])
            log_prob_id = kde.score_samples(id_unc[:, None])
            log_prob_ood = kde.score_samples(ood_unc[:, None])

            # Threshold for accuracy calculation
            threshold = np.percentile(log_prob_id, 5)
            id_pred = log_prob_id > threshold
            ood_pred = log_prob_ood > threshold

            y_true = np.concatenate([np.ones(len(log_prob_id)), np.zeros(len(log_prob_ood))])
            y_pred = np.concatenate([id_pred, ood_pred])
            scores = np.concatenate([log_prob_id, log_prob_ood])
            from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
            # Metrics
            acc = accuracy_score(y_true, y_pred)
            auroc = roc_auc_score(y_true, scores)
            aupr = average_precision_score(y_true, scores)

            results.append({
                "metric": metric,
                "dice_bin": bin_id,
                "acc": acc,
                "auroc": auroc,
                "aupr": aupr,
                "n_id": len(id_unc),
                "n_ood": len(ood_unc),
            })

            print(f"Bin {bin_id}: ACC={acc:.3f}, AUROC={auroc:.3f}, AUPR={aupr:.3f}")

        # Convert to DataFrame for easy analysis
        results_df = pd.DataFrame(results)

        # Map raw metric column names to pretty names
        metric_pretty_map = {
            "epkl_pred_mean_unc": "EPKL",
            "confidence_pred_mean_unc": "Confidence",
            "entropy_pred_mean_unc": "Entropy",
            "mutual-info_pred_mean_unc": "Mutual Info"
        }

        # Map dice bins to their ranges for captions
        dice_bin_caption = {
            0: "0.0–0.1",
            1: "0.1–0.5",
            2: "0.5–0.7",
            3: ">0.7"
        }

        # Apply the mapping
        results_df['metric_pretty'] = results_df['metric'].map(metric_pretty_map)
        results_df['dice_caption'] = results_df['dice_bin'].map(dice_bin_caption)

        import matplotlib.pyplot as plt
        import seaborn as sns
        # Lineplot with dice bins on x-axis and captions underneath
        plt.figure(figsize=(8, 6))
        sns.lineplot(
            data=results_df,
            x="dice_bin",
            y="auroc",
            hue="metric_pretty",
            marker="o"
        )

        # Set x-ticks to the dice bins, with captions underneath
        plt.xticks(ticks=list(dice_bin_caption.keys()),
                   labels=[f"{b}\n({dice_bin_caption[b]})" for b in dice_bin_caption])

        plt.title("AUROC per Dice Bin per Metric")
        plt.xlabel("Dice Bin")
        plt.ylabel("AUROC")
        plt.ylim(0.0, 1.05)
        plt.grid(True)
        plt.legend(title="Metric")
        plt.tight_layout()
        plt.savefig("/gpfs/home6/palfken/ood_features/auroc_per_bin_pretty.png")
        plt.show()


        #
        # # Assuming you have results_df from before
        # plt.figure(figsize=(8, 6))
        # sns.lineplot(data=results_df, x="dice_bin", y="auroc", hue="metric", marker="o")
        # plt.title("AUROC per Dice Bin per Metric")
        # plt.xlabel("Dice Bin")
        # plt.ylabel("AUROC")
        # plt.legend(title="Metric")
        # plt.ylim(0.0, 1.05)
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig("/gpfs/home6/palfken/ood_features/auroc_per_bin.png")
        # plt.show()

    #
    # for idx, metric in enumerate(unc_cols_pred):
    #     print(f'METRICS FOR {metric}')
    #     for bin in range(4):
    #         print(f'{bin}: {metric}')
    #         unc_col = unc_cols_pred[idx]
    #         unc_col_ood = unc_cols_pred_ood[idx]
    #
    #         # Extract and clean ID values
    #         id_unc = df[unc_col].values
    #         print("NaNs in ID:", np.isnan(id_unc).sum())
    #         id_unc_clean = id_unc[~np.isnan(id_unc)]
    #
    #         # Extract and clean OOD values
    #         ood_unc = df_ood[unc_col_ood].values
    #         print("NaNs in OOD before cleaning:", np.isnan(ood_unc).sum())
    #         ood_unc_clean = ood_unc[~np.isnan(ood_unc)]
    #         print("NaNs in OOD after cleaning:", np.isnan(ood_unc_clean).sum())
    #
    #         # Define quantile thresholds from ID
    #         q95 = np.quantile(id_unc_clean, 0.95)
    #         q98 = np.quantile(id_unc_clean, 0.98)
    #         print(f"95th percentile (ID): {q95:.4f}")
    #         print(f"98th percentile (ID): {q98:.4f}")
    #
    #         # Check how many OOD points fall above thresholds
    #         frac_ood_above95 = np.mean(ood_unc_clean > q95)
    #         frac_ood_above98 = np.mean(ood_unc_clean > q98)
    #         frac_id_above95 = np.mean(id_unc_clean > q95)  # should be ~5%
    #         frac_id_above98 = np.mean(id_unc_clean > q98)  # should be ~2%
    #
    #         print(f"OOD above 95th: {frac_ood_above95:.2%}, ID above 95th: {frac_id_above95:.2%}")
    #         print(f"OOD above 98th: {frac_ood_above98:.2%}, ID above 98th: {frac_id_above98:.2%}")
    #
    #         # Plot distributions
    #         plt.hist(id_unc_clean, bins=30, alpha=0.6, label='ID')
    #         plt.hist(ood_unc_clean, bins=30, alpha=0.6, label='OOD')
    #         plt.axvline(q95, color='red', linestyle='--', label='95th ID')
    #         plt.axvline(q98, color='black', linestyle='--', label='98th ID')
    #         plt.xlabel('Uncertainty')
    #         plt.ylabel('Count')
    #         plt.title(f'{metric} Uncertainty distribution of ID Validation vs OOD Lipoma Set with 95/98% cutoff')
    #         plt.legend()
    #         plt.show()
    #         plt.savefig(f"/gpfs/home6/palfken/ood_features/unc_thresholds_{metric}.png")
    #         plt.close()
    #
    #         kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(id_unc_clean[:, None])
    #         log_prob = kde.score_samples(ood_unc_clean[:, None])
    #         log_prob_id = kde.score_samples(id_unc[:, None])
    #
    #         threshold = np.percentile(log_prob_id, 5)  # bottom 5% likelihood of ID
    #         id_pred = log_prob_id > threshold
    #         ood_pred = log_prob > threshold
    #
    #         # Build ground truth + preds
    #         y_true = np.concatenate([np.ones(len(log_prob_id)), np.zeros(len(log_prob))])
    #         y_pred = np.concatenate([id_pred, ood_pred])
    #
    #         from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
    #         # Accuracy
    #         acc = accuracy_score(y_true, y_pred)
    #         print(f"Accuracy (5% cutoff): {acc:.3f}")
    #
    #         # AUROC (better measure, uses raw log probs)
    #         scores = np.concatenate([log_prob_id, log_prob])
    #         auroc = roc_auc_score(y_true, scores)
    #         print(f"AUROC: {auroc:.3f}")
    #
    #         # ---- Visualization ----
    #         plt.figure(figsize=(8, 5))
    #         plt.scatter(range(len(log_prob_id)), log_prob_id, label="ID", alpha=0.6)
    #         plt.scatter(range(len(log_prob_id), len(log_prob_id) + len(log_prob)), log_prob, label="OOD", alpha=0.6,
    #                 color="orange")
    #         plt.axhline(threshold, color="red", linestyle="--", label="5% threshold (ID)")
    #         plt.xlabel("Sample index")
    #         plt.ylabel("Log-likelihood under ID KDE")
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.title(f"Log-likelihood under ID KDE with {metric}")
    #         plt.show()
    #         plt.savefig(f"/gpfs/home6/palfken/ood_features/log_likelihood_{metric}.png")
    #         plt.close()

        # plt.hist(log_prob_id, bins=30, alpha=0.6, label='ID')
        # plt.hist(log_prob, bins=30, alpha=0.6, label='OOD')
        # plt.xlabel('Log-Likelihood under ID KDE')
        # plt.ylabel('Count')
        # plt.legend()
        # plt.savefig(f"/gpfs/home6/palfken/log_prob_{metric}.png")
        # plt.close()
        #
        # from sklearn.metrics import roc_auc_score, average_precision_score
        # y_true = np.concatenate([np.zeros(len(id_unc)), np.ones(len(ood_unc_clean))])
        # scores = np.concatenate([id_unc, ood_unc_clean])
        #
        # roc = roc_auc_score(y_true, scores)
        # pr = average_precision_score(y_true, scores)
        #
        # print(f"AUROC: {roc:.3f}, AUPR: {pr:.3f}")
        #
        # # --- Combine for plotting ---
        # df_plot = pd.concat([
        #     pd.DataFrame({'uncertainty': df[unc_col], 'dice_bin': df['dice_bin'], 'dist': 'In-Distribution'}),
        #     pd.DataFrame(
        #         {'uncertainty': df_ood[unc_col_ood], 'dice_bin': df_ood['dice_bin'], 'dist': 'OOD'})
        # ])
        #
        # # --- Plot ---
        # plt.figure(figsize=(10, 6))
        # sns.violinplot(x='dice_bin', y='uncertainty', hue='dist', data=df_plot, split=True, inner='quartile',
        #                palette='Set2')
        # plt.title(f'Predicted Mean {metrics[idx]} by Dice Bin')
        # plt.xlabel('Dice Bin')
        # plt.ylabel('Predicted Mean Uncertainty')
        # plt.legend(title='Distribution')
        # plt.tight_layout()
        # plt.savefig(f"/gpfs/home6/palfken/violin_{metric}.png")
        # plt.close()
        #
        # plt.figure(figsize=(12, 6))
        #
        # for i, bin_label in enumerate(dice_bins):
        #     plt.subplot(1, len(dice_bins), i + 1)
        #
        #     id_unc = df.loc[df['dice_bin'] == bin_label, unc_col]
        #     ood_unc = df_ood.loc[df_ood['dice_bin'] == bin_label, unc_col_ood]
        #
        #     # Smooth KDE plots
        #     sns.kdeplot(id_unc, fill=True, alpha=0.4, color=colors['In-Distribution'], label='ID')
        #     sns.kdeplot(ood_unc, fill=True, alpha=0.4, color=colors['OOD'], label='OOD')
        #
        #     # Mean lines
        #     plt.axvline(id_unc.mean(), color=colors['In-Distribution'], linestyle='--')
        #     plt.axvline(ood_unc.mean(), color=colors['OOD'], linestyle='--')
        #
        #     plt.title(f'Dice Bin: {bin_label}')
        #     plt.xlabel(f'Predicted Mean {metrics[idx]}')
        #     plt.ylabel('Density')
        #     if i == 0:
        #         plt.legend()
        #
        # plt.tight_layout()
        # plt.savefig(f"/gpfs/home6/palfken/hist_kde_{metric}.png")
        # plt.close()

    # for col in unc_cols_pred:
    #     means = df.groupby('dice_bin')[col].mean()
    #     stds = df.groupby('dice_bin')[col].std()
    #
    #     plt.errorbar(means.index, means.values, yerr=stds.values, label=col, capsize=5, marker='o')
    #
    # plt.xlabel('Dice bin')
    # plt.ylabel('Mean Uncertainty ± std')
    # plt.title('Uncertainty metrics by Dice bins')
    # plt.legend()
    # plt.xticks(ticks=[0, 1, 2, 3], labels=['<=0.1', '0.1-0.5', '0.5-0.7', '>0.7'])
    # plt.savefig("/gpfs/home6/palfken/mean_uncertainty.png")
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


if __name__ == '__main__':
    main()

