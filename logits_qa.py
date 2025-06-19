"""
The Logits-based Segmentation Quality Assessment Baseline

Logits are extracted from right before Softmax is applied in the nnunetv2 to get the final prediction.

Logits from nnUNet output should have shape [C,D,H,W]
"""
import numpy as np
import nibabel as nib
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr


preds_dir = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/LogitsTs'
image_dir = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnUNet_raw/Dataset002_SoftTissue/labelsTs'
tabular_data_dir = r'/home/bmep/plalfken/my-scratch/Downloads/WORC_data/WORC_all_with_nnunet_ids.csv'
tabular_data = pd.read_csv(tabular_data_dir)
table_dir = r'/home/bmep/plalfken/my-scratch/Downloads/logit_features.csv'
df = pd.read_csv(table_dir)
#print(tabular_data.head())
subtype = tabular_data[['nnunet_id','Final_Classification']]
#
#
def crop_to_min_shape(arr1, arr2):
    min_shape = tuple(min(a, b) for a, b in zip(arr1.shape, arr2.shape))
    return arr1[tuple(slice(0, s) for s in min_shape)], arr2[tuple(slice(0, s) for s in min_shape)]

def compute_confidence_score(logits, pred_mask, class_of_interest=1):
    """
    Compute mean max softmax probability over predicted tumor region.

    Args:
        logits: torch.Tensor or np.ndarray of shape (C, Z, Y, X)
        pred_mask: np.ndarray or torch.Tensor binary mask of predicted tumor (1 inside tumor, else 0)
        class_of_interest: int, class index for tumor (usually 1)

    Returns:
        mean_confidence: float - average confidence inside predicted tumor region
    """
    if isinstance(logits, np.ndarray):
        #logits = np.clip(logits, -1e3, 1e3)
        #print("Clipped Logits")
        #print(np.min(logits), np.max(logits))
        logits = torch.from_numpy(logits)


    if isinstance(pred_mask, np.ndarray):
        pred_mask = torch.from_numpy(pred_mask).bool()
    else:
        pred_mask = pred_mask.bool()
    logits = logits.clamp(min=-100, max=100)

    probs = F.softmax(logits, dim=0)  # softmax over classes
    tumor_probs = probs[class_of_interest]  # shape (X, Y, Z)

    if pred_mask.sum() == 0:
        return float('nan')
    pred_mask_reshaped = np.transpose(pred_mask, (2,1,0))

    if tumor_probs.shape != pred_mask_reshaped.shape:
        pred_mask_reshaped, tumor_probs = crop_to_min_shape(pred_mask_reshaped, tumor_probs)

    # Index tumor_probs with pred_mask without any transpose/permute
    mean_confidence = tumor_probs[pred_mask_reshaped].mean().item()


    return mean_confidence

def correlate_confidence_dice(confidences, dices):
    """
    Compute Pearson and Spearman correlation between confidence scores and dice scores.

    Args:
        confidences: list or np.ndarray of floats
        dices: list or np.ndarray of floats

    Returns:
        dict with pearson and spearman correlation coefficients and p-values
    """
    # Remove nan values (if any)
    valid_idx = ~np.isnan(confidences)
    confidences = np.array(confidences)[valid_idx]
    dices = np.array(dices)[valid_idx]

    pearson_corr, pearson_p = pearsonr(confidences, dices)
    spearman_corr, spearman_p = spearmanr(confidences, dices)

    return {
        'pearson_corr': pearson_corr,
        'pearson_pval': pearson_p,
        'spearman_corr': spearman_corr,
        'spearman_pval': spearman_p
    }



def compute_msp(logits, axis = 0):
    """
    Compute Maximum Softmax Probability (MSP)
    High MSP -> High confidence
    """
    logits = logits - np.max(logits, axis=axis, keepdims=True)  # stability trick
    e_x = np.exp(logits)
    probs = e_x / np.sum(e_x, axis = axis, keepdims = True)
    return probs.max(axis =axis), probs



def compute_dice(pred, gt):
    pred = np.asarray(pred == 1, dtype =np.uint8)
    gt= (np.asarray(gt == 1, dtype=np.uint8))

    intersection = np.sum(pred* gt) #true positives
    union = np.sum(pred) + np.sum(gt) #predicted plus ground truth voxels

    dice = 2. * intersection / union if union > 0 else np.nan

    return dice



def compute_logit_gap(logits):
    """
       Compute the difference between top-1 and top-2 logits at each voxel.
       Returns: mean and std of logit gaps.
       """

    #print(f"Before clipping: min={logits.min()}, max={logits.max()}")
    logits = np.clip(logits, -1e5, 1e5)
    #print(f"After clipping: min={logits.min()}, max={logits.max()}")
    sorted_logits = np.sort(logits, axis=0)  # sort over class axis (C)
    gap = sorted_logits[-1, ...] - sorted_logits[-2, ...]
    gap = gap.astype(np.float64)

    # Compute std over spatial dimensions (all voxels)
    # print(f"Gap stats before nan_to_num:")
    # print(f"  min={gap.min()}, max={gap.max()}, mean={gap.mean()}")
    # print(f"  any nan={np.isnan(gap).any()}, any inf={np.isinf(gap).any()}")

    logit_gap = np.nan_to_num(gap, nan=0.0, posinf=1e6, neginf=-1e6)

    # print(f"Gap stats after nan_to_num:")
    # print(f"  min={gap.min()}, max={gap.max()}, mean={gap.mean()}")
    # print(f"  any nan={np.isnan(gap).any()}, any inf={np.isinf(gap).any()}")

    #logit_gap_std = np.std(logit_gap, axis=(0, 1, 2))  # This collapses all spatial dims into one number
    logit_gap_std = np.std(logit_gap)
    sample_gap = gap.flatten()[:10000]

    logit_gap_std_small = np.std(sample_gap)
    #print(f'STD on smaller sample : {logit_gap_std_small}')
    # Similarly mean if needed
    logit_gap_mean = np.mean(logit_gap, axis=(0, 1, 2))
    return logit_gap_std, logit_gap_mean

def collect_features():
    df = pd.DataFrame(columns=["case_id", "subtype", "dice_score","min_msp","mean_msp", "conf","logit_gap_mean", "logit_gap_std"])

    for file in os.listdir(preds_dir):
        if file.endswith('.nii.gz'):

            stem = file.replace('.nii.gz', '')
            print(f'Processing {stem}')
            # === retrieve patient data ===
            mask_dir = os.path.join(preds_dir, file)
            gt_dir = os.path.join(image_dir, stem + '.nii.gz')
            logits_dir = os.path.join(preds_dir, f'{stem}_resampled_logits.npy.npz')
            #print(f'Ground truth directory {gt_dir}')
            #print(f'Mask directory: {mask_dir}')
            #print(f'Logits directory: {logits_dir}')
            logits1 = np.load(logits_dir)

            logits = logits1[logits1.files[0]]
            logits_reshaped = np.transpose(logits, (0,3, 2, 1))
            print(logits.dtype)

            print('Logits shape:', logits_reshaped.shape)

            mask = nib.load(mask_dir).get_fdata()

            print(f'Mask shape {mask.shape}')
            gt = nib.load(gt_dir).get_fdata()
            print(f'GT shape {gt.shape}')
            mask = mask.astype(np.uint8)
            patient = subtype[subtype['nnunet_id'] == stem]
            if not patient.empty:
                subtype_label = patient['Final_Classification'].values[0]
            else:
                subtype_label = "Unknown"
            print(f'Subtype {subtype_label}')
            #print(f'Mask shape: {mask.shape}, GT shape: {gt.shape}, Logits shape: {logits.shape}')

            # === compute all features ===
            dice = compute_dice(mask, gt)
            print(f'Dice: {dice}')

            msp_map, probs = compute_msp(logits, axis=0)
            min_msp = np.min(msp_map)
            mean_msp = np.mean(msp_map)




            # === Logit gap ===
            logit_gap_std, logit_gap_mean  = compute_logit_gap(logits)


            conf = compute_confidence_score(logits, mask)
            print(f'Confidence: {conf}')
            #print(f'Logit gap std: {logit_gap_std}')

            df.loc[len(df)] = [
                stem, subtype_label, dice, min_msp,
                mean_msp, conf, logit_gap_mean, logit_gap_std
            ]


    print(f"Processed {len(df)} cases")
    return df

def get_conf():
    conf_list =[]
    case_ids = []
    subtypes = []
    dice_scores = []
    for file in os.listdir(preds_dir):
        if file.endswith('.nii.gz'):

            stem = file.replace('.nii.gz', '')
            print(f'Processing {stem}')
            # === retrieve patient data ===
            mask_dir = os.path.join(preds_dir, file)
            gt_dir = os.path.join(image_dir, stem + '.nii.gz')
            logits_dir = os.path.join(preds_dir, f'{stem}_resampled_logits.npz')
            #print(f'Ground truth directory {gt_dir}')
            #print(f'Mask directory: {mask_dir}')
            #print(f'Logits directory: {logits_dir}')
            logits1 = np.load(logits_dir)

            logits = logits1[logits1.files[0]]
            logits_reshaped = np.transpose(logits, (0,3, 2, 1))

            patient = subtype[subtype['nnunet_id'] == stem]
            if not patient.empty:
                subtype_label = patient['Final_Classification'].values[0]
            else:
                subtype_label = "Unknown"
            print(f'Subtype {subtype_label}')
            subtypes.append(subtype_label)
            #print(f'Mask shape: {mask.shape}, GT shape: {gt.shape}, Logits shape: {logits.shape}')

            mask = nib.load(mask_dir).get_fdata()


            gt = nib.load(gt_dir).get_fdata()

            mask = mask.astype(np.uint8)

            # === compute all features ===
            dice = compute_dice(mask, gt)
            print(f'Dice: {dice}')
            dice_scores.append(dice)

            conf = compute_confidence_score(logits, mask)
            print(f'Confidence: {conf}')


            case_ids.append(stem)
            conf_list.append(conf)

    return pd.DataFrame({
        "case_id": case_ids,
        "subtype": subtypes,
        "dice_score": dice_scores,
        "conf": conf_list
    })






def correlate_metrics_per_subtype(df, metrics, target_col='dice_score', group_col='subtype'):
    """
    Computes Pearson and Spearman correlations of multiple metrics with Dice, per subtype.

    Args:
        df: pandas DataFrame with all columns
        metrics: list of metric column names to correlate with target
        target_col: the target column (e.g., 'dice_score')
        group_col: column to group by (e.g., 'subtype')

    Returns:
        pandas DataFrame with correlations per metric per subtype
    """
    results = []
    for subtype, group in df.groupby(group_col):
        for metric in metrics:
            x = group[metric].values
            y = group[target_col].values
            valid = ~np.isnan(x) & ~np.isnan(y)
            if valid.sum() > 1:
                pearson_corr, pearson_p = pearsonr(x[valid], y[valid])
                spearman_corr, spearman_p = spearmanr(x[valid], y[valid])
                results.append({
                    'subtype': subtype,
                    'metric': metric,
                    'pearson_corr': pearson_corr,
                    'pearson_pval': pearson_p,
                    'spearman_corr': spearman_corr,
                    'spearman_pval': spearman_p,
                    'n_samples': valid.sum()
                })
            else:
                results.append({
                    'subtype': subtype,
                    'metric': metric,
                    'pearson_corr': np.nan,
                    'pearson_pval': np.nan,
                    'spearman_corr': np.nan,
                    'spearman_pval': np.nan,
                    'n_samples': valid.sum()
                })
    return pd.DataFrame(results)



def create_corr_map():
    numeric_df = df.drop(columns=['case_id'])
    df_encoded = pd.get_dummies(numeric_df, columns=['subtype'])


    # Compute correlation matrix
    correlation_matrix = df_encoded.corr()['dice_score'].sort_values(ascending=False)

    print("Correlation of each feature with Dice score:\n")
    print(correlation_matrix)
    # Extract correlation between confidence metrics and Dice
   # dice_corr = correlation_matrix[['dice_score']].T

    # Choose only confidence metrics and subtype one-hots
    columns_to_plot = ['dice_score', 'min_msp', 'mean_msp', 'logit_gap_mean', 'logit_gap_std']
    corr = df_encoded[columns_to_plot].corr()
    print(corr.columns)

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr[['dice_score']].sort_values(by='dice_score', ascending=False), annot=True, cmap='coolwarm')
    plt.title('Correlation of Confidence Scores with Dice')
    plt.tight_layout()
    plt.show()




# table_dir = r'/home/bmep/plalfken/my-scratch/Downloads/logit_features.csv'
# os.makedirs(os.path.dirname(table_dir), exist_ok=True)
# df.to_csv(table_dir, index=False)
#print(df.head(13))
#print(df['logit_gap_std'].describe())
#print(df.isna().sum())


# data = collect_features()
# print(data.head(10))


#df_final.drop(['conf_x', 'conf_y'])
# = get_conf()
#conf_scores.to_csv(r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/conf_val.csv')
conf_scores = pd.read_csv(r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/conf_train.csv')
df_clean = conf_scores.dropna(subset=["dice_score", "conf", "subtype"])
results = []

# Loop through subtypes
for subtype in df_clean["subtype"].unique():
    subset = df_clean[df_clean["subtype"] == subtype]

    if len(subset) < 3:
        print(f"Skipping {subtype} (not enough samples)")
        continue

    dice = subset["dice_score"].values
    conf = subset["conf"].values
    # Compute Pearson and Spearman correlations
    pearson_corr, _ = pearsonr(conf, dice)
    spearman_corr, _ = spearmanr(conf, dice)

    results.append({
        "subtype": subtype,
        "n": len(subset),
        "pearson": round(pearson_corr, 3),
        "spearman": round(spearman_corr, 3),
        "mean_dice": round(dice.mean(), 3),
        "std_dice": round(dice.std(), 3),
        "mean_conf": round(conf.mean(), 3),
        "std_conf": round(conf.std(), 3),

    })

correlation_df = pd.DataFrame(results)
print(correlation_df.sort_values(by="pearson", ascending=False))

# metrics= ['min_msp', 'mean_msp', 'logit_gap_mean', 'logit_gap_std','conf']
# results = correlate_metrics_per_subtype(df_final,metrics)
# df.to_csv(r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/Logits_qa.csv', index = False)
#
# sns.set_theme(style="whitegrid")
# plt.figure(figsize=(12, 6))
# sns.barplot(data=results, x="subtype", y="pearson_corr", hue="metric")
# plt.xticks(rotation=45)
# plt.title("Per-Subtype Pearson Correlation with Dice Score")
# plt.ylabel("Pearson Correlation")
# plt.tight_layout()
# plt.show()
'''
dice_score                    1.000000
subtype_LeiomyoSarcomas       0.273008
subtype_WDLPS                 0.232021
subtype_MyxofibroSarcomas     0.054252
subtype_DTF                  -0.167174
mean_msp                     -0.287034
subtype_MyxoidlipoSarcoma    -0.299917
logit_gap_std                -0.522346
logit_gap_mean               -0.536154
min_msp                      -0.882409
'''