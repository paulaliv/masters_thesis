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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch.optim as optim


preds_dir = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/Logits'
image_dir = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnUNet_raw/Dataset002_SoftTissue/labelsTs'
tabular_data_dir = r'/home/bmep/plalfken/my-scratch/Downloads/WORC_data/WORC_all_with_nnunet_ids.csv'
tabular_data = pd.read_csv(tabular_data_dir)
table_dir = r'/home/bmep/plalfken/my-scratch/Downloads/logit_features.csv'
df = pd.read_csv(table_dir)
#print(tabular_data.head())
subtype = tabular_data[['nnunet_id','Final_Classification']]

class TemperatureScaler(torch.nn.Module):
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits/ self.temperature

def learn_temperature(logits, labels):
    logits = logits.detach()  # ensure logits don't require grad
    labels = labels.detach()
    # Instantiate scaler
    scaler = TemperatureScaler()

    # Optimizer to learn temperature
    optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=100)

    # Cross-entropy loss
    loss_fn = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        scaled_logits = scaler(logits)
        loss = loss_fn(scaled_logits, labels)
        loss.backward()
        return loss

    # Optimize temperature
    optimizer.step(closure)

    print(f"Learned temperature: {scaler.temperature.item():.4f}")
    return scaler.temperature.item()

def compute_msp(logits, axis = 0):
    """
    Compute Maximum Softmax Probability (MSP)
    High MSP -> High confidence
    """
    logits = logits - np.max(logits, axis=axis, keepdims=True)  # stability trick
    e_x = np.exp(logits)
    probs = e_x / np.sum(e_x, axis = axis, keepdims = True)
    return probs.max(axis =axis), probs


# def compute_entropy(probs, axis = 0):
#     """Compute voxel-wise entropy from softmax probabilities."""
#     eps = 1e-8
#     probs = np.clip(probs, eps, 1.0)
#     # avoid zeros
#     entropy = -np.sum(probs * np.log(probs), axis=axis)
#
#
#     return entropy

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
    df = pd.DataFrame(columns=["case_id", "subtype", "dice_score","min_msp","mean_msp",
        "logit_gap_mean", "logit_gap_std"])

    for file in os.listdir(preds_dir):
        if file.endswith('.nii.gz'):

            stem = file.replace('.nii.gz', '')
            print(f'Processing {stem}')
            # === retrieve patient data ===
            mask_dir = os.path.join(preds_dir, file)
            gt_dir = os.path.join(image_dir, stem + '.nii.gz')
            logits_dir = os.path.join(preds_dir, f'{stem}_resampled_logits.npy')
            #print(f'Ground truth directory {gt_dir}')
            #print(f'Mask directory: {mask_dir}')
            #print(f'Logits directory: {logits_dir}')
            logits = np.load(logits_dir)
            mask = nib.load(mask_dir).get_fdata()
            gt = nib.load(gt_dir).get_fdata()
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
            #print(f'Logit gap std: {logit_gap_std}')

            df.loc[len(df)] = [
                stem, subtype_label, dice, min_msp,
                mean_msp, logit_gap_mean, logit_gap_std
            ]


    print(f"Processed {len(df)} cases")
    return df

def run_regression_with_min_msp(X_train,X_test,y_train,y_test):

    model_min_msp = LinearRegression()
    model_min_msp.fit(X_train, y_train)

    # Predict
    y_pred_min_msp = model_min_msp.predict(X_test)

    # Metrics
    print('Model 1 - min_msp only')
    print('MSE:', mean_squared_error(y_test, y_pred_min_msp))
    print('R2:', r2_score(y_test, y_pred_min_msp))


def create_corr_map():
    numeric_df = df.drop(columns=['case_id'])
    df_encoded = pd.get_dummies(numeric_df, columns=['subtype'])


    # Compute correlation matrix
    correlation_matrix = df_encoded.corr()['dice_score'].sort_values(ascending=False)

    print("Correlation of each feature with Dice score:\n")
    print(correlation_matrix)
    # Extract correlation between confidence metrics and Dice
    dice_corr = correlation_matrix[['dice_score']].T

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


logits_list = []
labels_list = []
counter = 0
batch_size = 3
for file in os.listdir(preds_dir):
    if file.endswith('.nii.gz'):

        stem = file.replace('.nii.gz', '')
        print(f'Processing {stem}')
        # === retrieve patient data ===
        mask_dir = os.path.join(preds_dir, file)
        gt_dir = os.path.join(image_dir, stem + '.nii.gz')

        logits_dir = os.path.join(preds_dir, f'{stem}_resampled_logits.npy')

        logits = np.load(logits_dir)

        logits = logits.reshape(logits.shape[0], -1)  # Flatten spatial dimensions → [C, N]
        #logits_mean = np.mean(logits, axis=1)  # Mean over spatial dims → [C]
        logits_list.append(logits.T)  # [N_voxels, C]

        #mask = nib.load(mask_dir).get_fdata()
        gt = nib.load(gt_dir).get_fdata()
        labels = gt.flatten().astype(int)  # [N_voxels]

        labels_list.append(labels)
        counter += 1
        if counter % batch_size == 0 or file == sorted(os.listdir(preds_dir))[-1]:
            batch_logits = np.concatenate(logits_list, axis=0)
            batch_labels = np.concatenate(labels_list, axis=0)

            logits_torch = torch.tensor(batch_logits, dtype=torch.float32)
            labels_torch = torch.tensor(batch_labels, dtype=torch.long)

            print(f'Running temperature scaling on batch of {counter} cases...')
            learn_temperature(logits_torch, labels_torch)

            logits_list.clear()
            labels_list.clear()






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