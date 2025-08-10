from collections import defaultdict
import gzip
from torch.cpu.amp import autocast
from sklearn.metrics import classification_report
import collections
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch.nn as nn
import torch
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import torch.nn.functional as F
import time
import sys
import json
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import seaborn as sns
import umap
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.amp import GradScaler, autocast
import random
import torch.optim as optim
from collections import Counter
#metrics:  MAE, MSE, RMSE, Pearson Correlation, Spearman Correlation
#Top-K Error: rank segmentation by quality (for human review)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from monai.transforms import MapTransform

class PrintKeysShape(MapTransform):
    def __call__(self, data):
        for key in self.keys:
            print(f"{key} shape: {data[key].shape}")
        return data

from monai.transforms import (
    Compose, EnsureChannelFirstd, EnsureTyped,RandRotate90d, RandFlipd,
    ToTensord, ConcatItemsd
)

train_transforms = Compose([
    RandRotate90d(keys=["image","mask", "uncertainty"], prob=0.5, max_k=3, spatial_axes=(1, 2)),

    RandFlipd(keys=["image","mask", "uncertainty"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image","mask", "uncertainty"], prob=0.5, spatial_axis=1),  # flip along height axis
    RandFlipd(keys=["image","mask", "uncertainty"], prob=0.5, spatial_axis=2),
    EnsureTyped(keys=["image","mask", "uncertainty"], dtype=torch.float32),

    ConcatItemsd(keys=["image", "mask"], name="image"),

    ToTensord(keys=["image", "uncertainty"])
])


val_transforms = Compose([
    EnsureTyped(keys=["image", "mask", "uncertainty"], dtype=torch.float32),
    ConcatItemsd(keys=["image", "mask"], name="image"),
    ToTensord(keys=["image", "uncertainty"])
])


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

class Light3DEncoder(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.MaxPool3d(2),  # <- NEW BLOCK
            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)) # outputs [B, 64, 1, 1, 1]
            # nn.AdaptiveAvgPool3d(1),
        )
        # self.encoder = nn.Sequential(
        #     nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
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

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)  # Flatten to [B, 64]


class QAModel(nn.Module):
    def __init__(self,num_thresholds):
        super().__init__()

        self.encoder_img = Light3DEncoder(in_channels=2)
        self.encoder_unc= Light3DEncoder(in_channels=1)
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


    def forward(self, image, uncertainty):
        x1 = self.encoder_img(image)
        x2 = self.encoder_unc(uncertainty)
        merged = torch.cat((x1, x2), dim=1) #[B,128]
        merged = self.norm(merged)
        features = self.fc(merged)  # [B, 1]

        #logits = features + self.biases  # Broadcast to [B, num_thresholds]
        return features


    def extract_features(self, uncertainty):
        x = self.encoder_unc(uncertainty)
        return x.view(x.size(0), -1)


def bin_dice_score(dice):
    # bin_edges = [0.1, 0.5, 0.7]  # 4 bins
    # return np.digitize(dice, bin_edges, right=False)
    epsilon = 1e-8
    dice = np.asarray(dice)
    dice_adjusted = dice - epsilon  # Shift slightly left
    bin_edges = [0.1, 0.5, 0.7]  # Same as before
    return np.digitize(dice_adjusted, bin_edges, right=True)  # right=True = (a <= x)


class QADataset(Dataset):
    def __init__(self, case_ids, data_dir, df, uncertainty_metric,transform=None, want_features = False):
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


        image = np.load(os.path.join(self.data_dir, f'{case_id}_img.npy'))
        mask = np.load(os.path.join(self.data_dir, f'{case_id}_pred.npy'))

        uncertainty = np.load(os.path.join(self.data_dir, f'{case_id}_{self.uncertainty_metric}.npy'))

        # Map dice score to category

        label = bin_dice_score(dice_score)

        label_tensor = torch.tensor(label).long()

        if self.transform:
            data = self.transform({
                "image": np.expand_dims(image, 0),
                'mask':np.expand_dims(mask, 0),
                "uncertainty": np.expand_dims(uncertainty, 0),
            })
            image = data["image"]
            uncertainty= data["uncertainty"]

        if self.want_features:
            return uncertainty, case_id, label_tensor
        else:
            return image, uncertainty, label_tensor, subtype

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

def decode_predictions(logits):
    # print("Logits mean:", logits.mean().item())
    # print("Logits min/max:", logits.min().item(), logits.max().item())
    #probs = torch.sigmoid(logits)

    return (logits > 0).sum(dim=1)


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


@torch.no_grad()
def corn_predict(logits):
    #logits shape: [B, num_thresholds]
    probs = torch.stack([-logits, logits], dim=2)  # shape: [B, num_thresholds, 2]
    pred = probs.softmax(dim=2).argmax(dim=2)  # [B, num_thresholds], values in {0,1}
    return pred.sum(dim=1)  # sum of positive threshold decisions = predicted class

def train_one_fold(fold,data_dir, df, splits, uncertainty_metric,plot_dir, device):
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

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,pin_memory=True)



    # Initialize your QA model and optimizer
    print('Initiating Model')
    model = QAModel(num_thresholds=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',  # 'min' if you want to reduce LR when monitored metric stops decreasing
                                  factor=0.1,  # factor by which the LR will be reduced. new_lr = old_lr * factor
                                  patience=5,  # number of epochs with no improvement after which LR will be reduced
                                  verbose=True,  # print messages when LR is reduced
                                  min_lr=1e-6,  # lower bound on the learning rate
                                  cooldown=0)

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    # warmup_scheduler = LambdaLR(optimizer, lr_lambda)

    #criterion = nn.BCEWithLogitsLoss()
    criterion = CORNLoss()

    #Early stopping variables
    best_val_loss = float('inf')
    patience = 10
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

    for epoch in range(60):
        print(f"Epoch {epoch + 1}/{60}")
        running_loss, correct, total = 0.0, 0, 0

        model.train()

        for image, uncertainty, label, _ in train_loader:
            label_counts = Counter(label.cpu().numpy().tolist())
            image, uncertainty, label = image.to(device),uncertainty.to(device), label.to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):

                preds = model(image, uncertainty)  # shape: [B, 3]

                #targets = encode_ordinal_targets(label).to(preds.device)
                #print(f'Tagets shape: {targets.shape}')
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
        val_preds_list.clear()
        val_labels_list.clear()
        val_subtypes_list.clear()

        print(f'Train label distribution: {label_counts}')


        with torch.no_grad():
            for image, uncertainty, label, subtype in val_loader:
                image, uncertainty, label = image.to(device), uncertainty.to(device), label.to(device)


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


        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total

        for param_group in optimizer.param_groups:
            print(f"Current LR: {param_group['lr']}")


        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        val_losses.append(epoch_val_loss)
        # avg_val_loss = sum(val_losses) / len(val_losses)

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

        scheduler.step(epoch_val_loss)


        #print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        # After each validation epoch:
        if kappa_quadratic > best_kappa:
            best_kappa = kappa_quadratic
            best_kappa_quad = kappa_quadratic
            best_kappa_lin = kappa_linear

            present_labels = np.unique(np.concatenate((val_labels_np, val_preds_np)))
            labels_idx = sorted([label for label in [0, 1, 2, 3] if label in present_labels])
            best_kappa_cm = confusion_matrix(val_labels_np, val_preds_np, labels=labels_idx)
            best_kappa_preds = val_preds_np.copy()
            best_kappa_labels = val_labels_np.copy()

            best_kappa_epoch = epoch


            #with gzip.open(f"model_fold{fold}_{uncertainty_metric}_ALL.pt.gz", 'wb') as f:
                torch.save(model.state_dict(), f, pickle_protocol=4)


        # Early stopping check
        if epoch_val_loss < best_val_loss:
            print(f'Yay, new best Val Loss: {epoch_val_loss}!')
            best_val_loss = epoch_val_loss
            patience_counter = 0


        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")

                break
    file = os.path.join(plot_dir, f'best_report_{uncertainty_metric}')
    with open(file, "w") as f:
        f.write(f"Final Classification Report for Fold {fold}:\n")
        f.write(best_report)

    # Plot and save best confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(best_kappa_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Best Confusion Matrix (Epoch {best_kappa_epoch}, κ² = {best_kappa:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"best_conf_matrix_fold{fold}_{uncertainty_metric}_ALL.png"))
    plt.close()

    print(f'Best Kappa of {best_kappa}observed after {best_kappa_epoch} epochs!')
    return train_losses, val_losses, best_kappa_preds, best_kappa_labels, best_kappa_quad, best_kappa_lin


def pad_to_max_length(loss_lists):
    max_len = max(len(lst) for lst in loss_lists)
    padded = []
    for lst in loss_lists:
        padded.append(np.pad(lst, (0, max_len - len(lst)), constant_values=np.nan))
    return np.array(padded)



def main(data_dir, plot_dir, folds,df):
    print('MODEL INPUT: UNCERTAINTY MAP + MASK WITH IMAGE')
    print('FUSION: LATE and EARLY for mask with image')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics = ['confidence', 'entropy', 'mutual_info', 'epkl']

    for idx, metric in enumerate(metrics):
        # Lists to aggregate results across folds
        all_val_preds = []
        all_val_labels = []

        all_train_losses = []
        all_val_losses = []
        all_kappas_quad = []
        all_kappas_lin = []
        start = time.time()
        for fold in range(5):

            train_losses, val_losses, val_preds, val_labels, best_kappa_quad, best_kappa_lin = train_one_fold(
                fold,
                data_dir=data_dir,
                df=df,
                splits=folds,
                uncertainty_metric=metric,
                plot_dir=plot_dir,
                device=device
            )
            # Aggregate per fold
            all_val_preds.append(val_preds)
            all_val_labels.append(val_labels)
            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)
            all_kappas_quad.append(best_kappa_quad)
            all_kappas_lin.append(best_kappa_lin)

        end = time.time()

        print(f"Total training time: {(end - start) / 60:.2f} minutes")

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
        plt.savefig(os.path.join(plot_dir, f'loss_curves_all_folds_{metric}_ALL.png'))
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
        plt.savefig(os.path.join(plot_dir, f"best_conf_matrix_all_folds_{metric}_ALL.png"))
        plt.close()



def plot_UMAP(train, y_train, model_array, neighbours, m, name, image_dir):
    print(f'feature shape {train.shape}')

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=neighbours,
        min_dist=0.1,
        metric=m,
        random_state=42
    )
    train_umap = reducer.fit_transform(train)  # (N, 2)
    # Apply UMAP transform to validation data
    # val_umap = reducer.transform(val)

    # all_subtypes= np.concatenate([y_train, y_val])
    unique_labels= sorted(set(y_train))

    # labels = np.array(['train'] * len(train_umap) + ['val'] * len(val_umap))
    markers = {'5': 'o', '20': 's', '30':'v'}

    idx_to_label = {
        '5':'Model trained for 5 epochs',
        '20':'Model trained for 20 epochs',
        '30':'Model trained for 30 epochs',

    }
    #class_names = ["Fail (0-0.1)", "Poor (0.1-0.5)", "Moderate(0.5-0.7)", " Good (>0.7)"]
    bin_to_score = {
        0: "Fail (0-0.1)",
        1: "Poor (0.1-0.5)",
        2: "Moderate(0.5-0.7)",
        3: "Good (>0.7)"

    }

    cmap = plt.cm.tab20
    color_lookup = {lab: cmap(i % 20) for i, lab in enumerate(unique_labels)}
    # 7. scatter plot
    plt.figure(figsize=(8, 6))
    for marker_type in ['5', '20', '30']:
        for label in unique_labels:
            idx = [i for i in range(len(y_train)) if y_train[i] == label and model_array[i] == marker_type]
            if not idx:
                continue
            plt.scatter(
                train_umap[idx, 0], train_umap[idx, 1],
                s=25,
                c=[color_lookup[label]] * len(idx),
                marker=markers[marker_type],
                alpha=0.8
            )

    # Create separate legends for color (dice bin) and shape (model type)
    color_handles = [
        mpatches.Patch(color=color_lookup[lab], label=bin_to_score[lab])
        for lab in unique_labels
    ]

    marker_handles = [
        mlines.Line2D([], [], color='black', marker=markers[key], linestyle='None',
                      markersize=6, label=idx_to_label[key])
        for key in markers
    ]

    # Plot legends
    legend1 = plt.legend(handles=color_handles, title='Dice Quality Bin', loc='upper right')
    plt.gca().add_artist(legend1)
    plt.legend(handles=marker_handles, title='Model Type', loc='lower right')

    plt.xlabel("UMAP‑1")
    plt.ylabel("UMAP‑2")
    plt.title("Confidence Map Features")
    plt.tight_layout()
    image_loc = os.path.join(image_dir, name)
    plt.savefig(image_loc, dpi=300)
    plt.show()
def visualize_features(data_dir, plot_dir, splits, df):
    # Create datasets

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QAModel(num_thresholds=3).to(device)
    uncertainty_metric = "confidence"
    train_case_ids = splits[0]["train"]
    train_dataset = QADataset(
        case_ids=train_case_ids,
        data_dir=data_dir,
        df=df,
        uncertainty_metric=uncertainty_metric,
        transform=train_transforms,
        want_features=True
    )
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,pin_memory=True)

    all_features_train = []
    all_labels_train = []
    all_model_names = []

    with torch.no_grad():
        for uncertainty, case_ids,labels in train_loader:
            uncertainty = uncertainty.to(device)
            features = model.extract_features(uncertainty).cpu().numpy()  # (B, 512)
            all_features_train.append(features)

            batch_labels = []
            for cid in case_ids:
                if "20EP" in cid:
                    model_ep = "20"
                elif "30EP" in cid:
                    model_ep = "30"
                else:
                    model_ep = "5"
                batch_labels.append(model_ep)

            all_model_names.extend(batch_labels)
            all_labels_train.extend(labels)



    X_train = np.concatenate(all_features_train, axis=0)
    y_train = np.array(all_labels_train)
    from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

    # X_train: (N, 512)
    distance_matrix = cosine_distances(X_train)  # or use euclidean_distances(X_train)
    model_array= np.array(all_model_names)

    # Ignore diagonal
    np.fill_diagonal(distance_matrix, np.inf)

    # Get top-k similar pairs (e.g., top-5 most similar pairs)
    k = 5
    similar_pairs = []
    for i in range(len(distance_matrix)):
        closest_indices = np.argsort(distance_matrix[i])[:k]
        for j in closest_indices:
            similar_pairs.append((i, j, distance_matrix[i][j]))

    # Sort by distance
    similar_pairs.sort(key=lambda x: x[2])
    for i1, i2, dist in similar_pairs[:30]:
        print(
            f"Pair: {train_case_ids[i1]} - {train_case_ids[i2]},  Distance: {dist:.4f}")

    plot_UMAP(X_train, y_train, model_array, neighbours=5, m='cosine', name='QA_UMAP_cosine_5n_fold0.png', image_dir=plot_dir)
    plot_UMAP(X_train, y_train,model_array, neighbours=10, m='cosine', name='QA_UMAP_cosine_10n_fold0.png', image_dir=plot_dir)
    plot_UMAP(X_train, y_train, model_array,neighbours=15, m='cosine', name='QA_UMAP_cosine_15n_fold0.png', image_dir=plot_dir)




if __name__ == '__main__':

    with open('/gpfs/home6/palfken/masters_thesis/Final_splits30.json', 'r') as f:
        splits = json.load(f)
    clinical_data = "/gpfs/home6/palfken/masters_thesis/Final_dice_dist1.csv"
    df =  pd.read_csv(clinical_data)

    preprocessed= sys.argv[1]
    plot_dir = sys.argv[2]

    main(preprocessed, plot_dir, splits, df)
    #visualize_features(preprocessed, plot_dir, splits, df)
