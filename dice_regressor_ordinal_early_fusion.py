from collections import defaultdict
from torch.optim.lr_scheduler import LambdaLR
from torch.cpu.amp import autocast
from sklearn.metrics import classification_report
import collections
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
import time
import sys
import json
from sklearn.metrics import cohen_kappa_score

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.amp import GradScaler, autocast
import random
import torch.optim as optim
#metrics:  MAE, MSE, RMSE, Pearson Correlation, Spearman Correlation
#Top-K Error: rank segmentation by quality (for human review)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, RandRotate90d,RandFlipd,RandAffined, RandGaussianNoised, NormalizeIntensityd,
    ToTensord, EnsureTyped, ConcatItemsd
)
train_transforms = Compose([


    RandRotate90d(keys=["mask", "uncertainty"], prob=0.5, max_k=3, spatial_axes=(1, 2)),

    RandFlipd(keys=["mask", "uncertainty"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["mask", "uncertainty"], prob=0.5, spatial_axis=1),  # flip along height axis
    RandFlipd(keys=["mask", "uncertainty"], prob=0.5, spatial_axis=2),
    EnsureTyped(keys=["mask", "uncertainty"], dtype=torch.float32),
    ConcatItemsd(keys=["mask", "uncertainty"], name="merged"),
    ToTensord(keys=["merged"])

])
val_transforms = Compose([
    EnsureTyped(keys=["mask", "uncertainty"], dtype=torch.float32),
    ConcatItemsd(keys=["mask", "uncertainty"], name="merged"),
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
        #     nn.Conv3d(2, 16, kernel_size=3, padding=1),
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
            nn.Conv3d(2, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.MaxPool3d(2),  # <- NEW BLOCK
            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))  # outputs [B, 64, 1, 1, 1]
            # nn.AdaptiveAvgPool3d(1),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)  # Flatten to [B, 64]


class QAModel(nn.Module):
    def __init__(self,num_thresholds):
        super().__init__()
        self.encoder = Light3DEncoder()
        #self.encoder_unc= Light3DEncoder()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_thresholds)  # Output = predicted Dice class
        )

    def forward(self, merged):

        x = self.encoder(merged)

        return self.fc(x)

def bin_dice_score(dice):
    bin_edges = [0.0, 0.1, 0.5, 0.7, 1.0]  # 6 bins
    label = np.digitize(dice, bin_edges, right=False) - 1
    return min(label, len(bin_edges) - 2)  # ensures label is in [0, 5]


class QADataset(Dataset):
    def __init__(self, case_ids, data_dir, df, uncertainty_metric, transform=None):
        """
        fold: str, e.g. 'fold_0'
        preprocessed_dir: base preprocessed path with .npz images
        pred_fold_paths: dict with predicted mask folder paths per fold
        fold_paths: dict with fold folder paths containing Dice scores & case IDs
        """
        #self.fold = fold
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
        #print(f"Fetching index {idx} out of {len(self.dice_scores)}")
        case_id = self.case_ids[idx]
        #print(case_id)
        dice_score = self.dice_scores[idx]
        #print(dice_score)
        subtype = self.subtypes[idx]
        subtype = subtype.strip()


        mask = np.load(os.path.join(self.data_dir, f'{case_id}_pred.npy'))
        uncertainty = np.load(os.path.join(self.data_dir, f'{case_id}_{self.uncertainty_metric}.npy'))


        # Map dice score to category
        label = bin_dice_score(dice_score)
        label_tensor = torch.tensor(label).long()


        if self.transform:
            data = self.transform({
                'mask': np.expand_dims(mask, 0),
                "uncertainty": np.expand_dims(uncertainty, 0),
            })
            merged = data["merged"]


        return  merged, label_tensor, subtype

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

def pad_collate_fn(batch):
    max_d = max(item[0].shape[1] for item in batch)
    max_h = max(item[0].shape[2] for item in batch)
    max_w = max(item[0].shape[3] for item in batch)

    images = []

    labels = []
    subtypes = []

    for image,label, subtype in batch:
        pad_dims = (
            0, max_w - image.shape[3],
            0, max_h - image.shape[2],
            0, max_d - image.shape[1]
        )
        image = torch.nn.functional.pad(image, pad_dims)

        images.append(image)
        labels.append(label)
        subtypes.append(subtype)

    images_batch = torch.stack(images)
    labels_batch = torch.stack(labels)  # works even if labels are scalar tensors
    # case_ids usually strings, so keep as list

    return images_batch,labels_batch, subtypes

def coral_loss_manual(logits, levels):
    """
    logits: [B, num_classes - 1]
    levels: binary cumulative targets (e.g., [1, 1, 0])
    """
    log_probs = F.logsigmoid(logits)
    log_1_minus_probs = F.logsigmoid(-logits)

    loss = -levels * log_probs - (1 - levels) * log_1_minus_probs
    return loss.mean()


def encode_ordinal_targets(labels, num_thresholds= 3): #K-1 thresholds
    batch_size = labels.shape[0]
    targets = torch.zeros((batch_size, num_thresholds), dtype=torch.float32)
    for i in range(num_thresholds):
        targets[:,i] = (labels > i).float()
    return targets


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

# def decode_predictions(logits):
#     probs = torch.sigmoid(logits)
#
#     return (probs > 0.5).sum(dim=1)

def train_one_fold(fold,data_dir, df, splits, uncertainty_metric, plot_dir, device):
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
    )
    val_dataset = QADataset(
        case_ids=val_case_ids,
        data_dir=data_dir,
        df=df,
        uncertainty_metric=uncertainty_metric,
        transform=val_transforms,
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,pin_memory=True,
    collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,pin_memory=True,
    collate_fn=pad_collate_fn)



    # Initialize your QA model and optimizer
    print('Initiating Model')
    model = QAModel(num_thresholds=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=5, verbose=True)

    # Optional: define class names for nicer output
    class_names = ["Fail (0-0.1)", "Poor (0.1-0.5)", "Moderate(0.5-0.7)", "Good (>0.7)"]
    # Step 4: Create the weighted loss
    #criterion = nn.BCEWithLogitsLoss()
    criterion = CORNLoss()

    # Define warmup parameters
    warmup_epochs = 5  # or warmup_steps if you're doing per-step

    # Linear warmup lambda
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / warmup_epochs
        return 1.0  # Once warmup is over, keep LR constant until ReduceLROnPlateau kicks in

    warmup_scheduler = LambdaLR(optimizer, lr_lambda)

    # Early stopping variables
    #DOUBLE CHECK THIS
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    scaler = GradScaler()

    train_losses = []
    val_losses = []

    f1_history = defaultdict(list)

    val_preds_list, val_labels_list, val_subtypes_list = [], [], []
    val_per_class_acc = defaultdict(list)

    best_kappa = -1.0
    best_kappa_cm = None
    best_kappa_epoch = -1

    for epoch in range(60):
        print(f"Epoch {epoch + 1}/{60}")
        running_loss, correct, total = 0.0, 0, 0

        model.train()

        for data,label, _ in train_loader:

            data, label = data.to(device),label.to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                preds = model(data)  # shape: [B, 3]
                #targets = encode_ordinal_targets(label).to(preds.device)
                loss = criterion(preds, label)


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * data.size(0)
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



        with torch.no_grad():
            for data,label, subtype in val_loader:
                data, label = data.to(device), label.to(device)


                preds = model(data)
                #targets = encode_ordinal_targets(label).to(preds.device)

                loss = criterion(preds, label)
                val_running_loss += loss.item() * data.size(0)

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


        scheduler.step(epoch_val_loss)
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
        for class_name in class_names:
            f1_history[class_name].append(report_dict[class_name]["f1-score"])

        #print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        if kappa_quadratic > best_kappa:
            best_kappa = kappa_quadratic
            best_kappa_quad = kappa_quadratic
            best_kappa_lin = kappa_linear
            labels_idx = [0, 1, 2, 3]
            #best_kappa_cm = confusion_matrix(val_labels_np, val_preds_np, labels=labels_idx)
            best_kappa_preds = val_preds_np.copy()
            best_kappa_labels = val_labels_np.copy()
            best_kappa_report = report
            best_kappa_epoch = epoch
        # Early stopping check


        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_report = classification_report(val_labels_np, val_preds_np, target_names=class_names, digits=4,
                                           zero_division=0)

        else:
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
    plt.title(f"Best Confusion Matrix (Epoch {best_kappa_epoch}, κ² = {best_kappa:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"best_conf_matrix_fold{fold}_{uncertainty_metric}_EARLY.png"))
    plt.close()

    print(f'Best Kappa of {best_kappa}observed after {best_kappa_epoch} epochs!')
    return train_losses, val_losses, best_kappa_preds, best_kappa_labels, best_kappa_quad, best_kappa_lin


# def compute_dice(pred, gt):
#     pred = np.asarray(pred == 1, dtype =np.uint8)
#     gt= (np.asarray(gt == 1, dtype=np.uint8))
#
#     intersection = np.sum(pred* gt) #true positives
#     union = np.sum(pred) + np.sum(gt) #predicted plus ground truth voxels
#
#     dice = 2. * intersection / union if union > 0 else np.nan
#
#     return dice
def pad_to_max_length(loss_lists):
    max_len = max(len(lst) for lst in loss_lists)
    padded = []
    for lst in loss_lists:
        padded.append(np.pad(lst, (0, max_len - len(lst)), constant_values=np.nan))
    return np.array(padded)

def main(data_dir, plot_dir, folds,df):
    print('MODEL INPUT: UNCERTAINTY MAP + IMAGE')
    print('FUSION: EARLY')

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
        plt.savefig(os.path.join(plot_dir, f'loss_curves_all_folds_{metric}_EARLY.png'))
        plt.close()

        # confusion matrix
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
        plt.savefig(os.path.join(plot_dir, f"best_conf_matrix_all_folds_{metric}_EARLY.png"))
        plt.close()

#metrics: confidence, entropy,mutual_info,epkl

if __name__ == '__main__':

    with open('/gpfs/home6/palfken/masters_thesis/Final_splits30.json', 'r') as f:
        splits = json.load(f)
    clinical_data = "/gpfs/home6/palfken/masters_thesis/Final_dice_dist1.csv"
    df =  pd.read_csv(clinical_data)

    preprocessed= sys.argv[1]
    plot_dir = sys.argv[2]

    main(preprocessed, plot_dir, splits, df)
