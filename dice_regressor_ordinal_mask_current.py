
import gzip
from torch.cpu.amp import autocast
from sklearn.metrics import classification_report
import collections
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


from itertools import product
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import torch.nn.functional as F
import time
import sys
import json
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import seaborn as sns
import umap
import plotly.graph_objects as go
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import seaborn as sns

from sklearn.decomposition import PCA
from torch.amp import GradScaler, autocast
import random
import torch.optim as optim
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F

#metrics:  MAE, MSE, RMSE, Pearson Correlation, Spearman Correlation
#Top-K Error: rank segmentation by quality (for human review)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# '''ARCHITECTURE OF THE INSPO PAPER'''
# class Light3DEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv3d(1, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool3d(2),
#
#             nn.Conv3d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool3d(2),
#
#             nn.Conv3d(64, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool3d(2),
#
#             nn.Conv3d(32, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool3d(2),
#
#             nn.Conv3d(32, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool3d(1),
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         return x.view(x.size(0), -1)  # Flatten to [B, 16]
#
# class QAModel(nn.Module):
#     def __init__(self,num_thresholds):
#         super().__init__()
#
#         self.encoder_img = Light3DEncoder()
#         self.encoder_unc= Light3DEncoder()
#         self.pool = nn.AdaptiveAvgPool3d(1)
#         self.norm = nn.LayerNorm(32)
#
#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(32, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_thresholds)  # ordinal logits
#         )
#
#
#     def forward(self, image, uncertainty):
#         x1 = self.encoder_img(image)
#         x2 = self.encoder_unc(uncertainty)
#         merged = torch.cat((x1, x2), dim=1) #[B,128]
#         merged = self.norm(merged)
#         return self.fc(merged)
#
#     def extract_features(self, uncertainty):
#         x = self.encoder_unc(uncertainty)
#         return x.view(x.size(0), -1)
#


'''PREVIOUS ARCHITECTURE'''
# class Light3DEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv3d(1, 16, kernel_size=3, padding=1),
#             nn.BatchNorm3d(16),
#             nn.ReLU(),
#             nn.MaxPool3d(2),  # halves each dimension
#
#             nn.Conv3d(16, 32, kernel_size=3, padding=1),
#             nn.BatchNorm3d(32),
#             nn.ReLU(),
#             nn.MaxPool3d(2),
#
#             nn.Conv3d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm3d(64),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool3d((1, 1, 1)),  # outputs [B, 64, 1, 1, 1]
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         return x.view(x.size(0), -1)  # Flatten to [B, 64]
#
# class QAModel(nn.Module):
#     def __init__(self,num_thresholds):
#         super().__init__()
#
#         self.encoder_img = Light3DEncoder()
#         self.encoder_unc= Light3DEncoder()
#         self.pool = nn.AdaptiveAvgPool3d(1)
#         self.norm = nn.LayerNorm(128)
#
#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_thresholds)  # Output = predicted Dice class
#         )
#
#     def forward(self, image, uncertainty):
#         x1 = self.encoder_img(image)
#         x2 = self.encoder_unc(uncertainty)
#         merged = torch.cat((x1, x2), dim=1) #[B,128]
#         merged = self.norm(merged)
#         return self.fc(merged)
#
#     def extract_features(self, uncertainty):
#         x = self.encoder_unc(uncertainty)
#         return x.view(x.size(0), -1)
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, RandRotate90d,RandFlipd,RandAffined, RandGaussianNoised, NormalizeIntensityd,
    ToTensord, EnsureTyped
)
train_transforms = Compose([


    RandRotate90d(keys=["mask", "uncertainty"], prob=0.5, max_k=3, spatial_axes=(1, 2)),

    RandFlipd(keys=["mask", "uncertainty"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["mask", "uncertainty"], prob=0.5, spatial_axis=1),  # flip along height axis
    RandFlipd(keys=["mask", "uncertainty"], prob=0.5, spatial_axis=2),
    EnsureTyped(keys=["mask", "uncertainty"], dtype=torch.float32),
    ToTensord(keys=["mask", "uncertainty"])
])
val_transforms = Compose([
    EnsureTyped(keys=["mask", "uncertainty"], dtype=torch.float32),
    ToTensord(keys=["mask", "uncertainty"])
])



class DistanceAwareCORNLoss(nn.Module):
    def __init__(self, eps=1e-6, distance_power=0.5):
        super().__init__()
        self.eps = eps
        self.distance_power = distance_power  # use sqrt by default

    def forward(self, logits, labels):
        """
        logits: [B, K-1] - logits for ordinal thresholds
        labels: [B] - integer labels in {0, ..., K-1}
        """
        B, K_minus_1 = logits.shape

        # Binary label matrix: y_bin[b, k] = 1 if label[b] > k
        y_bin = torch.zeros_like(logits, dtype=torch.long)
        for k in range(K_minus_1):
            y_bin[:, k] = (labels > k).long()

        # Reshape logits for binary cross-entropy
        logits_stacked = torch.stack([-logits, logits], dim=2)  # [B, K-1, 2]
        logits_reshaped = logits_stacked.view(-1, 2)            # [B*(K-1), 2]
        targets_reshaped = y_bin.view(-1)                       # [B*(K-1)]

        # Compute distance-based weights (e.g., sqrt(|label - k|))
        label_expanded = labels.unsqueeze(1).expand(-1, K_minus_1)  # [B, K-1]
        ks = torch.arange(K_minus_1, device=labels.device).unsqueeze(0)  # [1, K-1]
        distances = torch.abs(label_expanded - ks).float()  # [B, K-1]
        weights = distances ** self.distance_power          # [B, K-1]

        # Normalize per-sample weights (optional but stabilizing)
        #weights = weights / (weights.sum(dim=1, keepdim=True) + self.eps)


        # Flatten for use in loss
        weights_flat = weights.view(-1)  # [B*(K-1)]

        # Compute weighted cross-entropy loss
        loss = F.cross_entropy(logits_reshaped, targets_reshaped, weight=None, reduction='none')
        loss = (loss * weights_flat).mean()

        return loss

class CORNLoss(nn.Module):
    """
    CORN Loss for ordinal regression.
    """
    def __init__(self):
        super(CORNLoss, self).__init__()

    def forward(self, logits, labels):
        """
        logits: Tensor of shape (B, K-1), where K is the number of ordinal classes
        labels: Tensor of shape (B,) with values in {0, ..., K-1}
        """
        B, K_minus_1 = logits.shape

        # temperature = 0.5
        #
        # logits = logits/temperature

        # Create binary targets: 1 if label > threshold
        y_bin = torch.zeros_like(logits, dtype=torch.long)
        for k in range(K_minus_1):
            y_bin[:, k] = (labels > k).long()

        # Compute softmax over two classes (not raw binary classification)
        # Each logit becomes a 2-class classification: [P(class <= k), P(class > k)]
        logits_stacked = torch.stack([-logits, logits], dim=2)  # shape: [B, K-1, 2]




        logits_reshaped = logits_stacked.view(-1, 2)  # [B*(K-1), 2]
        targets_reshaped = y_bin.view(-1)  # [B*(K-1)]

        loss = F.cross_entropy(logits_reshaped, targets_reshaped, reduction='mean')
        return loss

class Light3DEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.encoder = nn.Sequential(
        #     nn.Conv3d(1, 16, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(16),
        #     nn.ReLU(),
        #     nn.MaxPool3d(2),  # halves each dimension
        #
        #     nn.Conv3d(16, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(),
        #     nn.MaxPool3d(2),
        #
        #     nn.Conv3d(32, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(64),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool3d((1, 1, 1)),  # outputs [B, 64, 1, 1, 1]
        # )

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.MaxPool3d(2),  # <- NEW BLOCK
            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))  # outputs [B, 64, 1, 1, 1]
            #nn.AdaptiveAvgPool3d(1),
        )
        #Output would now be [B, 128]

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)  # Flatten to [B, 64]



class QAModel(nn.Module):
    def __init__(self,num_thresholds):
        super().__init__()

        self.encoder_img = Light3DEncoder()
        self.encoder_unc= Light3DEncoder()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.norm = nn.LayerNorm(256)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_thresholds)  # Output = predicted Dice class
        )
        #self.biases = nn.Parameter(torch.zeros(num_thresholds))


    def forward(self, mask, uncertainty):
        x1 = self.encoder_img(mask)
        x2 = self.encoder_unc(uncertainty)
        merged = torch.cat((x1, x2), dim=1) #[B,128]
        merged = self.norm(merged)
        features = self.fc(merged)  # [B, 1]

        #logits = features + self.biases  # Broadcast to [B, num_thresholds]
        return features

    def extract_unc_features(self, uncertainty):
        x = self.encoder_unc(uncertainty)
        return x.view(x.size(0), -1)


    def extract_mask_features(self, mask):
        x = self.encoder_img(mask)
        return x.view(x.size(0), -1)

def bin_dice_score(dice):
    # bin_edges = [0.1, 0.5, 0.7]  # 4 bins
    # return np.digitize(dice, bin_edges, right=False)
    epsilon = 1e-8
    dice = np.asarray(dice)
    dice_adjusted = dice - epsilon  # Shift slightly left
    bin_edges = [0.1, 0.5, 0.7]  # Same as before
    return np.digitize(dice_adjusted, bin_edges, right=True)  # right=True = (a <= x)


def compute_class_distribution(labels):
    """
    labels: list or 1D numpy array or tensor of ordinal class labels (e.g., from 0 to 3)
    """
    counter = Counter(labels)
    total = sum(counter.values())
    distribution = {k: v / total for k, v in sorted(counter.items())}
    return distribution



class QADataset(Dataset):
    def __init__(self, case_ids, data_dir, df, uncertainty_metric,transform=None, want_features = False, is_ood = False):
        """
        fold: str, e.g. 'fold_0'
        preprocessed_dir: base preprocessed path with .npz images
        pred_fold_paths: dict with predicted mask folder paths per fold
        fold_paths: dict with fold folder paths containing Dice scores & case IDs
        """
        #self.fold = fold
        self.want_features = want_features

        self.data_dir = data_dir
        self.df = df
        self.uncertainty_metric = uncertainty_metric


        # Load Dice scores & case IDs from a CSV or JSON
        # Example CSV: case_id,dice

        # List of case_ids
        self.case_ids = case_ids
        self.df = df.set_index('case_id').loc[self.case_ids].reset_index()


        # Now extract dice scores and subtypes aligned with self.case_ids
        if is_ood:
            self.dice_scores = self.df['dice'].tolist()
        else:
            self.dice_scores = self.df['dice_5'].tolist()

        self.subtypes = self.df['tumor_class'].tolist()

        self.transform = transform

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        dice_score = self.dice_scores[idx]
        subtype = self.subtypes[idx]
        subtype = subtype.strip()


        mask = np.load(os.path.join(self.data_dir, f'{case_id}_pred.npy'))
        #image = torch.from_numpy(image).float()


        uncertainty = np.load(os.path.join(self.data_dir, f'{case_id}_{self.uncertainty_metric}.npy'))


        # Map dice score to category

        label = bin_dice_score(dice_score)


        #image_tensor = torch.from_numpy(image).float()

        # print(f'Image shape {image.shape}')
        #uncertainty_tensor = torch.from_numpy(uncertainty).float()

        #uncertainty_tensor = uncertainty_tensor.unsqueeze(0)  # Add channel dim


        label_tensor = torch.tensor(label).long()

        if self.transform:
            data = self.transform({
                "mask": np.expand_dims(mask, 0),
                "uncertainty":np.expand_dims(uncertainty, 0),
            })
            mask= data["mask"].float()
            uncertainty = data["uncertainty"].float()


        if self.want_features:
            return mask, uncertainty,label_tensor, subtype, case_id
        else:
            return mask, uncertainty, label_tensor, subtype

def get_padded_shape(shape, multiple=16):
    return tuple(((s + multiple - 1) // multiple) * multiple for s in shape)

def pad_tensor(t, target_shape):
    """
    Pads a 4D tensor (C, D, H, W) to a target shape (D, H, W), only on the right/bottom sides.
    Returns a tensor of shape (C, D, H, W), padded with zeros.
    """


    if t.ndim == 4:
        t = t.unsqueeze(0)  # (1, C, D, H, W)

    _, C, D, H, W = t.shape
    target_D, target_H, target_W = target_shape

    pad = (
        0, target_W - W,  # pad width (W)
        0, target_H - H,  # pad height (H)
        0, target_D - D   # pad depth (D)
    )

    t = F.pad(t, pad)  # Now shape is (1, C, D', H', W')
    return t.squeeze(0)  # Return (C, D', H', W')


def encode_ordinal_targets(labels, num_thresholds= 3): #K-1 thresholds
    batch_size = labels.shape[0]
    targets = torch.zeros((batch_size, num_thresholds), dtype=torch.float32)
    for i in range(num_thresholds):
        targets[:,i] = (labels > i).float()
    return targets

#hard thresholding of 0.5 biased towoards predicting middle classes
#try: use argmax on cumulative logits
#or logit based decoding

# logit based decofing: (logits > 0).sum(dim=1)If all 3 logits > 0 → class 3
# If first 2 > 0, last ≤ 0 → class 2
# If only 1st > 0 → class 1
# If all ≤ 0 → class 0
#
# ✅ This works well but relies heavily on raw logit thresholds at zero, which may be unstable or biased.
def decode_predictions(logits):
    # print("Logits mean:", logits.mean().item())
    # print("Logits min/max:", logits.min().item(), logits.max().item())
    #probs = torch.sigmoid(logits)

    return (logits > 0).sum(dim=1)

@torch.no_grad()
def corn_predict(logits):
    #logits shape: [B, num_thresholds]
    probs = torch.stack([-logits, logits], dim=2)  # shape: [B, num_thresholds, 2]
    pred = probs.softmax(dim=2).argmax(dim=2)  # [B, num_thresholds], values in {0,1}
    return pred.sum(dim=1)  # sum of positive threshold decisions = predicted class

def evaluate_per_threshold(logits, labels):
    """
    logits: [N, K-1] - model outputs
    labels: [N] - ground truth labels (ordinal, from 0 to K-1)
    """
    N, K_minus_1 = logits.shape

    # Create binary targets per threshold
    y_bin = np.zeros_like(logits, dtype=np.int32)
    for k in range(K_minus_1):
        y_bin[:, k] = (labels > k).astype(int)

    metrics = {}
    for k in range(K_minus_1):
        prob = torch.sigmoid(logits[:, k]).cpu().numpy()
        true_bin = y_bin[:, k]

        pred_bin = (prob > 0.5).astype(int)

        acc = accuracy_score(true_bin, pred_bin)
        try:
            auc = roc_auc_score(true_bin, prob)
        except:
            auc = None  # In case only one class is present

        metrics[f"Threshold {k}"] = {
            "Accuracy": acc,
            "AUROC": auc
        }

    return metrics
def coral_loss_manual(logits, levels, smoothing = 0.2, entropy_weight = 0.01):
    """
    logits: [B, num_classes - 1]
    levels: binary cumulative targets (e.g., [1, 1, 0])
    """
    levels =levels.float()
    levels = levels * (1 - smoothing) + 0.5 * smoothing

    log_probs = F.logsigmoid(logits)
    log_1_minus_probs = F.logsigmoid(-logits)

    loss = -levels * log_probs - (1 - levels) * log_1_minus_probs


    #
    # importance_weights = torch.tensor([2, 3, 1.0])
    # importance_weights = importance_weights.to(logits.device)
    # importance_weights = importance_weights.view(1, -1)  # for broadcasting
    # loss = loss * importance_weights

    # Reduce loss across thresholds per sample, then average over batch
    loss = loss.sum(dim=1).mean()
    # probs = torch.sigmoid(logits)
    # entropy = -(probs * log_probs + (1 - probs) * log_1_minus_probs).mean()
    # loss += entropy_weight * entropy

    return loss


def train_one_fold(fold,data_dir, df, splits, uncertainty_metric,plot_dir, device, epochs,lre, batch_size, warmup_epochs, patience):
    print(f'Training with UMAP: {uncertainty_metric}')
    print(f"Training fold {fold} ...")

    train_case_ids = splits[fold]["train"]
    val_case_ids = splits[fold]["val"]

    # Create datasets
    train_dataset = QADataset(
        case_ids=train_case_ids,
        data_dir=data_dir,
        df=df,
        uncertainty_metric=uncertainty_metric,
        transform=train_transforms,
        want_features=False,
    )
    val_dataset = QADataset(
        case_ids=val_case_ids,
        data_dir=data_dir,
        df=df,
        uncertainty_metric=uncertainty_metric,
        transform=val_transforms,
        want_features=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)



    # Initialize your QA model and optimizer
    print('Initiating Model')
    model = QAModel(num_thresholds=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lre)



    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=5, verbose=True,min_lr=1e-6)
    # Step 1: Warmup

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    #
    # # Step 2: Cosine Annealing after warmup
    plateau_scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',  # 'min' if you want to reduce LR when monitored metric stops decreasing
                                  factor=0.1,  # factor by which the LR will be reduced. new_lr = old_lr * factor
                                  patience=5,  # number of epochs with no improvement after which LR will be reduced
                                  verbose=True,  # print messages when LR is reduced
                                  min_lr=1e-6,  # lower bound on the learning rate
                                  cooldown=0)

    #scheduler = CosineAnnealingLR(optimizer, T_max=35)  # 45 = total_epochs - warmup_epochs

    # Combine them
    #scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[5])
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    #criterion = nn.BCEWithLogitsLoss()
    #criterion = coral_loss_manual
    criterion = CORNLoss()

    #Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    #Initiate Scaler
    scaler = GradScaler()

    train_losses = []
    val_losses = []

    #Kappa variables
    best_kappa = -1.0
    best_kappa_cm = None
    best_kappa_epoch = -1

    val_preds_list, val_labels_list, val_subtypes_list = [], [], []

    class_names = ["Fail (0-0.1)", "Poor (0.1-0.5)", "Moderate(0.5-0.7)", " Good (>0.7)"]

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        running_loss, correct, total = 0.0, 0, 0

        model.train()

        for image, uncertainty, label, _ in train_loader:
            label_counts = Counter(label.cpu().numpy().tolist())
            image, uncertainty, label = image.to(device),uncertainty.to(device), label.to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                # print(f'Label shape {label.shape}')
                preds = model(image, uncertainty)  # shape: [B, 3]
                # print(f'model Output Shape : {preds.shape}')
                #targets = encode_ordinal_targets(label).to(preds.device)

                loss = criterion(preds, label)


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * image.size(0)
            total += label.size(0)

            with torch.no_grad():
                #decoded_preds = decode_predictions(preds)
                decoded_preds = corn_predict(preds)

                correct += (decoded_preds == label).sum().item()


        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")

        train_losses.append(epoch_train_loss)

        # Validation step
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0

        empty_mask_count = 0
        total_masks = 0

        all_logits = []
        all_labels = []

        val_preds_list.clear()
        val_labels_list.clear()
        val_subtypes_list.clear()


        with torch.no_grad():
            for image, uncertainty, label, subtype in val_loader:

                image, uncertainty, label = image.to(device), uncertainty.to(device), label.to(device)

                # Count empty masks in the batch
                batch_empty = (image.sum(dim=[1, 2, 3, 4]) == 0).sum().item()  # assumes mask shape is (B, C, D, H, W)
                empty_mask_count += batch_empty
                total_masks += image.size(0)

                preds = model(image, uncertainty)
                #targets = encode_ordinal_targets(label).to(preds.device)


                loss = criterion(preds, label)
                val_running_loss += loss.item() * image.size(0)



                with torch.no_grad():
                    #decoded_preds = decode_predictions(preds)
                    decoded_preds = corn_predict(preds)
                    val_correct += (decoded_preds == label).sum().item()


                val_total += label.size(0)

                val_preds_list.extend(decoded_preds.cpu().numpy())
                val_labels_list.extend(label.cpu().numpy())

                # Convert subtype to list
                if isinstance(subtype, torch.Tensor):
                    subtype_list = subtype.cpu().numpy().tolist()
                else:
                    subtype_list = subtype

                val_subtypes_list.extend(subtype_list)

        print(f"{empty_mask_count} out of {total_masks} masks in validation were completely empty.")

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total

        for param_group in optimizer.param_groups:
            print(f"Current LR: {param_group['lr']}")

        val_losses.append(epoch_val_loss)

        # all_logits = torch.cat(all_logits, dim=0)
        # all_labels = torch.cat(all_labels, dim=0).numpy()
        #
        # threshold_metrics = evaluate_per_threshold(all_logits, all_labels)
        #
        # for t, metric in threshold_metrics.items():
        #     print(f"{t} => Accuracy: {metric['Accuracy']:.3f}, AUROC: {metric['AUROC']}")
        # # avg_val_loss = sum(val_losses) / len(val_losses)

        val_preds_np = np.array(val_preds_list)
        val_labels_np = np.array(val_labels_list)

        # 'linear' or 'quadratic' weights are valid for ordinal tasks
        kappa_linear = cohen_kappa_score(val_labels_np, val_preds_np, weights='linear')
        kappa_quadratic = cohen_kappa_score(val_labels_np, val_preds_np, weights='quadratic')

        print("Linear Kappa:", kappa_linear)
        print("Quadratic Kappa:", kappa_quadratic)


        report = classification_report(val_labels_np, val_preds_np, target_names=class_names, digits=4, zero_division=0)

        print("Validation classification report:\n", report)
        print("Predicted label counts:", collections.Counter(val_preds_list))
        print("True label counts:", collections.Counter(val_labels_list))
        report_dict = classification_report(
            val_labels_np,
            val_preds_np,
            target_names=class_names,
            digits=4,
            output_dict=True,  # this is key
            zero_division=0
        )

        #
        # # Step the appropriate scheduler
        # if epoch < warmup_epochs:
        #     warmup_scheduler.step()
        #     print(f"[Warmup] LR: {optimizer.param_groups[0]['lr']:.6f}")
        if epoch < warmup_epochs:
            warmup_scheduler.step()  # warmup scheduler
            print(f"[Warm Up] LR: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            plateau_scheduler.step(epoch_val_loss)  # ReduceLROnPlateau scheduler

            print(f"[ReduceLROnPlateau] LR: {optimizer.param_groups[0]['lr']:.6f}")



        if kappa_quadratic > best_kappa:
            print(f'New Best Kappa: {kappa_quadratic}!')
            best_kappa = kappa_quadratic
            best_kappa_quad = kappa_quadratic
            best_kappa_lin = kappa_linear


            present_labels = np.unique(np.concatenate((val_labels_np, val_preds_np)))
            labels_idx = sorted([label for label in [0, 1, 2, 3] if label in present_labels])
            best_kappa_cm = confusion_matrix(val_labels_np, val_preds_np, labels=labels_idx)
            best_kappa_preds = val_preds_np.copy()
            best_kappa_labels = val_labels_np.copy()
            best_kappa_report = report
            best_kappa_epoch = epoch

            with gzip.open(f"model_fold{fold}_{uncertainty_metric}.pt.gz", 'wb') as f:
                torch.save(model.state_dict(), f, pickle_protocol=4)


        # Early stopping check
        if epoch_val_loss < best_val_loss:
            print(f'YAY, new best Val loss: {epoch_val_loss}!')
            best_val_loss = epoch_val_loss
            patience_counter = 0
            present_labels = np.unique(np.concatenate((val_labels_np, val_preds_np)))
            labels_idx = sorted([label for label in [0, 1, 2, 3] if label in present_labels])
            best_val_epoch = epoch
            best_val_cm = confusion_matrix(val_labels_np, val_preds_np, labels=labels_idx)

        else:
            print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")

                break

    # Plot and save best confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(best_kappa_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Best Confusion Matrix (Epoch {best_kappa_epoch}, κ = {best_kappa_lin:.3f}, κ² = {best_kappa_quad:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"best_conf_matrix_fold{fold}_{uncertainty_metric}_MASK.png"))
    plt.close()


    plt.figure(figsize=(6, 5))
    sns.heatmap(best_val_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Best Confusion Matrix (Epoch {best_val_epoch}, Validation loss = {best_val_loss:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"val_conf_matrix_fold{fold}_{uncertainty_metric}_MASK.png"))
    plt.close()

    print(f'Best Kappa of {best_kappa}observed after {best_kappa_epoch} epochs!')
    del train_loader, val_loader, train_dataset, val_dataset, model, optimizer
    torch.cuda.empty_cache()  # frees GPU memory

    return train_losses, val_losses, best_kappa_preds, best_kappa_labels,  best_kappa_quad, best_kappa_lin


def pad_to_max_length(loss_lists):
    max_len = max(len(lst) for lst in loss_lists)
    padded = []
    for lst in loss_lists:
        padded.append(np.pad(lst, (0, max_len - len(lst)), constant_values=np.nan))
    return np.array(padded)

def main(data_dir, plot_dir, folds,df):
    print('MODEL INPUT: UNCERTAINTY MAP AND MASK')
    print('FUSION: LATE')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #metrics = ['confidence', 'entropy','mutual_info','epkl']
    metrics = ['mutual_info']

    # param_grid = {
    #     'lr': [1e-3, 3e-4, 1e-4],
    #     'batch_size': [16, 32],
    #     'warmup_epochs': [3, 5, 8],
    #     'patience': [8, 10, 15],
    # }
    #
    #
    # best_params_per_metric = {}
    #
    for metric in metrics:
        print(f"\n=== Tuning for metric: {metric} ===")

    #     best_score = -float('inf')
    #     best_params = None
    #
    #     # Step 1: Tune on just fold 0
    #     for lr, bs, warmup, patience in product(
    #             param_grid['lr'], param_grid['batch_size'],
    #             param_grid['warmup_epochs'], param_grid['patience']
    #     ):
    #         print(f"Testing params: LR={lr}, BS={bs}, Warmup={warmup}, Patience={patience}")
    #
    #         train_losses, val_losses, val_preds, val_labels, kappa_quad, kappa_lin = train_one_fold(
    #             fold=0,  # tuning only on fold 0
    #             data_dir=data_dir,
    #             df=df,
    #             splits=folds,
    #             uncertainty_metric=metric,
    #             plot_dir=plot_dir,
    #             device=device,
    #             epochs=30,
    #             lre=lr,
    #             batch_size=bs,
    #             warmup_epochs=warmup,
    #             patience=patience
    #         )
    #
    #         if kappa_quad > best_score:
    #             best_score = kappa_quad
    #             best_params = {
    #                 'lr': lr,
    #                 'batch_size': bs,
    #                 'warmup_epochs': warmup,
    #                 'patience': patience
    #             }
    #
    #     print(f"Best params for {metric}: {best_params} (kappa={best_score:.4f})")
    #     best_params_per_metric[metric] = best_params

        # Step 2: Full 5-fold training with best params
        print(f"=== Running full CV for {metric} ===")

        all_val_preds = []
        all_val_labels = []

        all_train_losses = []
        all_val_losses = []
        all_kappas_quad = []
        all_kappas_lin = []
        start = time.time()
        for fold in range(5):
            train_losses, val_losses, val_preds, val_labels, kappa_quad, kappa_lin = train_one_fold(
                fold=fold,
                data_dir=data_dir,
                df=df,
                splits=folds,
                uncertainty_metric=metric,
                plot_dir=plot_dir,
                device=device,
                epochs=60,
                lre=1e-3,
                batch_size=16,
                warmup_epochs=3,
                patience=15
            )

            # Aggregate per fold
            all_val_preds.append(val_preds)
            all_val_labels.append(val_labels)
            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)
            all_kappas_quad.append(kappa_quad)
            all_kappas_lin.append(kappa_lin)


        end = time.time()
        # print(f'Best report for {metric}:')
        # print(best_report)
        print()
        print(f"Total training time: {(end - start) / 60:.2f} minutes")
        print()

        # Combine folds data
        val_preds = np.concatenate(all_val_preds, axis=0)
        val_labels = np.concatenate(all_val_labels, axis=0)

        padded_train_losses = pad_to_max_length(all_train_losses)
        avg_train_losses = np.nanmean(padded_train_losses, axis=0)

        padded_val_losses = pad_to_max_length(all_val_losses)
        avg_val_losses = np.nanmean(padded_val_losses, axis=0)



        avg_kappa_quad = np.mean(all_kappas_quad)
        print(f'Average Quadratic Kappa across all 5 folds: {avg_kappa_quad}')

        avg_kappa_lin = np.mean(all_kappas_lin)
        print(f'Average Linear Kappa across all 5 folds: {avg_kappa_lin}')

        plt.figure(figsize=(10, 6))
        plt.plot(avg_train_losses, label='Train Loss', marker='o')
        plt.plot(avg_val_losses, label='Validation Loss', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Training and Validation Loss Curves - {metric}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'loss_curves_all_folds_{metric}_MASK.png'))
        plt.close()

        # Overall scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(val_labels, val_preds, c='blue', alpha=0.6, label='Predicted vs Actual')
        plt.plot([val_labels.min(), val_labels.max()],
                 [val_labels.min(), val_labels.max()], 'r--', label='45-degree line')
        plt.xlabel("Actual Label")
        plt.ylabel("Predicted Label")
        plt.title(f"Overall Predicted vs. Actual Labels - {metric}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'pred_vs_actual_{metric}_MASK.png'))
        plt.close()

        #confusion matrix
        # Plot and save best confusion matrix
        class_names = ["Fail (0-0.1)", "Poor (0.1-0.5)", "Moderate(0.5-0.7)", " Good (>0.7)"]
        present_labels = np.unique(np.concatenate((val_labels, val_preds)))
        labels_idx = sorted([label for label in [0, 1, 2, 3] if label in present_labels])

        disp = confusion_matrix(val_labels, val_preds, labels=labels_idx)

        plt.figure(figsize=(6, 5))
        sns.heatmap(disp, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix: {metric}, (κ = {avg_kappa_lin:.3f}, κ² = {avg_kappa_quad:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"best_conf_matrix_all_folds_{metric}_MASK.png"))
        plt.close()

    # for metric in metrics:
    #     print(f"Best params for {metric}: {best_params_per_metric[metric] }")
    #


def plot_UMAP(X_val, y_val, subtypes_val, X_ood, y_ood, subtypes_ood, neighbours, m, name, image_dir):

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=neighbours,
        min_dist=0.1,
        metric=m,
        random_state=42
    )

    # Combine validation and OOD data
    X_combined = np.concatenate([X_val, X_ood])
    y_combined = np.concatenate([y_val, y_ood])  # dice bins
    subtypes_combined = np.concatenate([subtypes_val, subtypes_ood])  # tumor subtypes

    # Optional PCA if dimensions > 50
    if X_combined.shape[1] > 50:
        pca = PCA(n_components=50, random_state=42)
        X_combined = pca.fit_transform(X_combined)

    # Compute UMAP embedding
    embedding = reducer.fit_transform(X_combined)

    jitter_strength = 0.5  # tweak as needed
    embedding += np.random.normal(scale=jitter_strength, size=embedding.shape)

    # Map each subtype to either ID or OOD category
    # This dict is subtype -> ID/OOD label
    tumor_to_idood = {
        "MyxofibroSarcomas": "ID",
        "LeiomyoSarcomas": "ID",
        "DTF": "ID",
        "MyxoidlipoSarcoma": "ID",
        "WDLPS": "ID",
        "Lipoma": "OOD",
        # add other subtypes if needed
    }

    # Map subtype labels to ID/OOD using vectorized mapping
    dist = np.array([tumor_to_idood.get(st, 'OOD') for st in subtypes_combined])

    # Define markers for ID/OOD subtypes
    markers = {'ID': 'o', 'OOD': 's'}

    # Define colors for dice bins (different colors for dice quality bins)
    bin_to_score = {
        0: "Fail (0-0.1)",
        1: "Poor (0.1-0.5)",
        2: "Moderate (0.5-0.7)",
        3: "Good (>0.7)"
    }

    # Use a qualitative colormap for dice bins
    cmap = plt.cm.Set1  # can choose others like tab10, tab20, etc.
    unique_bins = sorted(set(y_combined))
    bin_to_color = {bin_val: cmap(i % cmap.N) for i, bin_val in enumerate(unique_bins)}

    plt.figure(figsize=(10, 8))

    # Plot each combination of dice_bin (color) and ID/OOD (marker)
    for dice_bin in unique_bins:
        for idood in ['ID', 'OOD']:
            idx = np.where((y_combined == dice_bin) & (dist == idood))[0]
            if len(idx) == 0:
                continue
            plt.scatter(
                embedding[idx, 0],
                embedding[idx, 1],
                c=[bin_to_color[dice_bin]] * len(idx),
                marker=markers[idood],
                label=f"{idood}-{bin_to_score[dice_bin]}",
                s=40,
                alpha=0.8,
                edgecolors='k'  # adds black border for visibility
            )

    # Build legend handles for dice bin colors
    color_handles = [
        mpatches.Patch(color=bin_to_color[bin_val], label=bin_to_score[bin_val])
        for bin_val in unique_bins
    ]

    # Build legend handles for markers (ID vs OOD)
    marker_handles = [
        mlines.Line2D([], [], color='black', marker=markers[idood], linestyle='None',
                      markersize=8, label=idood)
        for idood in markers
    ]

    legend1 = plt.legend(handles=color_handles, title='Dice Quality Bin', loc='upper right')
    plt.gca().add_artist(legend1)
    plt.legend(handles=marker_handles, title='Sample Type', loc='lower right')

    plt.xlabel("UMAP‑1")
    plt.ylabel("UMAP‑2")
    plt.title("UMAP of Dice QA Features (Dice Bins as Colors, ID vs OOD as Markers)")
    plt.tight_layout()

    os.makedirs(image_dir, exist_ok=True)
    image_loc = os.path.join(image_dir, name)
    plt.savefig(image_loc, dpi=300)
    plt.close()
    # Fit on validation + OOD combined

def plot_UMAP_1(X_val, y_val, neighbours, m, name, image_dir):

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=neighbours,
        min_dist=0.1,
        metric=m,
        random_state=42
    )


    # Optional PCA if dimensions > 50
    if X_val.shape[1] > 50:
        pca = PCA(n_components=50, random_state=42)
        X_val = pca.fit_transform(X_val)

    # Compute UMAP embedding
    embedding = reducer.fit_transform(X_val)

    jitter_strength = 0.5  # tweak as needed
    embedding +=  np.random.normal(scale=jitter_strength, size=embedding.shape)

    # Define colors for dice bins (different colors for dice quality bins)
    bin_to_score = {
        0: "Fail (0-0.1)",
        1: "Poor (0.1-0.5)",
        2: "Moderate (0.5-0.7)",
        3: "Good (>0.7)"
    }

    # Use a qualitative colormap for dice bins
    cmap = plt.cm.Set1  # can choose others like tab10, tab20, etc.
    unique_bins = sorted(set(y_val))
    bin_to_color = {bin_val: cmap(i % cmap.N) for i, bin_val in enumerate(unique_bins)}

    plt.figure(figsize=(10, 8))


    # Plot each dice_bin
    for dice_bin in unique_bins:
        idx = np.where(y_val == dice_bin)[0]
        if len(idx) == 0:
            continue
        plt.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            c=[bin_to_color[dice_bin]] * len(idx),
            label=f"{bin_to_score.get(dice_bin, dice_bin)}",
            s=40,
            alpha=0.8,
            edgecolors='k'
        )

    # Build legend handles for dice bin colors
    color_handles = [
        mpatches.Patch(color=bin_to_color[bin_val], label=bin_to_score[bin_val])
        for bin_val in unique_bins
    ]

    legend1 = plt.legend(handles=color_handles, title='Dice Quality Bin', loc='upper right')
    plt.gca().add_artist(legend1)
    plt.legend(handles=color_handles, title='Sample Type', loc='lower right')

    plt.xlabel("UMAP‑1")
    plt.ylabel("UMAP‑2")
    plt.title("UMAP of Dice QA Features (Dice Bins as Colors)")
    plt.tight_layout()

    os.makedirs(image_dir, exist_ok=True)
    image_loc = os.path.join(image_dir, name)
    plt.savefig(image_loc, dpi=300)
    plt.close()
    # Fit on validation + OOD combined



def inference(ood_dir, uncertainty_metric):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subtypes_csv = "/gpfs/home6/palfken/WORC_test.csv"
    subtypes_df = pd.read_csv(subtypes_csv)
    dice_dir ="/gpfs/home6/palfken/Dice_scores_OOD.csv"
    dice_df = pd.read_csv(dice_dir)

    lipoma_ids = subtypes_df.loc[
        subtypes_df["Final_Classification"] == "Lipoma",
        "nnunet_id"
    ].values

    # Filter dice_df to only Lipoma rows
    dice_df_lipoma = dice_df[dice_df["tumor_class"] == "Lipoma"]
    print(f"{len(lipoma_ids),len(dice_df_lipoma)} OOD cases found")

    ood_dataset = QADataset(
        case_ids=lipoma_ids,
        data_dir=ood_dir,
        df=dice_df_lipoma,
        uncertainty_metric=uncertainty_metric,
        transform=val_transforms,
        want_features=True,
        is_ood=True
    )
    ood_loader = DataLoader(ood_dataset, batch_size=4, shuffle=False, pin_memory=True)

    fold_paths = [
        f"/gpfs/home6/palfken/model_fold0_{uncertainty_metric}.pt.gz",
        f"/gpfs/home6/palfken/model_fold1_{uncertainty_metric}.pt.gz",
        f"/gpfs/home6/palfken/model_fold2_{uncertainty_metric}.pt.gz",
        f"/gpfs/home6/palfken/model_fold3_{uncertainty_metric}.pt.gz",
        f"/gpfs/home6/palfken/model_fold4_{uncertainty_metric}.pt.gz",

    ]


    all_subtypes = []
    all_preds = []
    all_labels = []
    all_case_ids = []


    for fold_idx, model_path in enumerate(fold_paths):
        with gzip.open(model_path, 'rb') as f:
            checkpoint = torch.load(f, map_location=device, weights_only=False)

        model = QAModel(num_thresholds=3).to(device)
        model.load_state_dict(checkpoint)
        model.eval()
        fold_preds = []

        with torch.no_grad():

            for mask, uncertainty, labels, subtype, case_id in ood_loader:
                mask, uncertainty =  mask.to(device), uncertainty.to(device)

                if fold_idx == 0:
                    all_subtypes.extend(subtype)
                    all_case_ids.extend(case_id)
                    all_labels.extend(labels.cpu().numpy())

                preds = model(mask, uncertainty).cpu()
                decoded_preds = corn_predict(preds)
                fold_preds.extend(decoded_preds)

        all_preds.append(fold_preds)



    all_case_ids = np.array(all_case_ids)
    all_subtypes = np.array(all_subtypes)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    from scipy.stats import mode
    final_preds, _ = mode(all_preds, axis=0)
    final_preds = final_preds.squeeze()
    #avg_preds= np.mean(all_preds, axis=0)

    return  {
            "case_ids": all_case_ids,
            "labels": all_labels,
            "subtypes": all_subtypes,
            "preds": final_preds
    }


def plot_confusion(y_true, y_pred, title, save_path):
    class_names = ["Fail (0-0.1)", "Poor (0.1-0.5)", "Moderate(0.5-0.7)", " Good (>0.7)"]
    present_labels = np.unique(y_pred)
    #labels_idx = sorted([label for label in [0, 1, 2, 3] if label in present_labels])
    cm = confusion_matrix(y_true, y_pred,  labels=[0, 1, 2, 3])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Bin")
    plt.ylabel("Actual Bin")
    plt.tight_layout()
    plt.title(title)

    plt.savefig(save_path, dpi=300)


def plot_bin_distribution(y_true, y_pred, title, save_path):
    bins = sorted(set(y_true) | set(y_pred))
    df = pd.DataFrame({
        "Actual": pd.Series(y_true).value_counts(normalize=True),
        "Predicted": pd.Series(y_pred).value_counts(normalize=True)
    }).fillna(0).reindex(bins)

    width = 0.35
    x = np.arange(len(bins))

    plt.bar(x - width/2, df["Actual"], width, label="Actual")
    plt.bar(x + width/2, df["Predicted"], width, label="Predicted")
    plt.xticks(x, bins)
    plt.ylabel("Proportion")
    plt.title(title)
    plt.legend()
    plt.savefig(save_path, dpi=300)




def visualize_features(ood_dir, plot_dir):


    metrics = ['confidence', 'entropy', 'mutual_info', 'epkl']
    for metric in metrics:
        results = inference(ood_dir=ood_dir,uncertainty_metric=metric)

        df = pd.DataFrame({
            "case_id": results["case_ids"],
            "gt": results["labels"],
            "subtype": results["subtypes"],
            "pred": results["preds"],
        })



        df.to_csv(os.path.join(plot_dir,f'{metric}_ood_results.csv'), index=False)


        # --- 2. Confusion matrix ---
        # Use val true labels and ood predicted labels (or vice versa depending on your setup)
        # Here I assume val labels vs ood labels as an example; adjust as needed
        plot_confusion(results["labels"],results["preds"],
                   title=f"Confusion Matrix - {metric} : WORC OOD",
                   save_path=os.path.join(plot_dir, f"confusion_ood_{metric}.png"))



if __name__ == '__main__':

    with open('/gpfs/home6/palfken/masters_thesis/Final_splits30.json', 'r') as f:
        splits = json.load(f)
    clinical_data = "/gpfs/home6/palfken/masters_thesis/Final_dice_dist1.csv"
    df =  pd.read_csv(clinical_data)

    #preprocessed= sys.argv[1]
    ood_dir = sys.argv[1]
    plot_dir = sys.argv[2]

    #main(preprocessed, plot_dir, splits, df)
    visualize_features( ood_dir, plot_dir)

