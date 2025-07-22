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
    Compose, EnsureChannelFirstd, RandFlipd,RandAffined, RandGaussianNoised,
    ToTensord
)
train_transforms = Compose([
    EnsureChannelFirstd(keys="merged", channel_dim=0),  # ensures shape: (C, H, W, D)
    RandAffined(
        keys="merged",
        prob=1.0,
        rotate_range=[np.pi / 9],
        translate_range=[0.1, 0.1],
        scale_range=[0.1, 0.1],
        mode='bilinear'  # single input, single interpolation mode
    ),
    RandFlipd(keys="merged", prob=0.5, spatial_axis=1),
    ToTensord(keys="merged")
])

val_transforms = Compose([
    EnsureChannelFirstd(keys="merged", channel_dim=0),
    ToTensord(keys="merged")
])


class Light3DEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),  # halves each dimension

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # outputs [B, 64, 1, 1, 1]
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
            nn.Linear(64, 64),
            nn.ReLU(),
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
        self.df = self.df[self.df['case_id'].isin(self.case_ids)]

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


        image = np.load(os.path.join(self.data_dir, f'{case_id}_img.npy'))
        uncertainty = np.load(os.path.join(self.data_dir, f'{case_id}_{self.uncertainty_metric}.npy'))

        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        uncertainty_tensor = torch.from_numpy(uncertainty).float().unsqueeze(0)

        # Map dice score to category
        label = bin_dice_score(dice_score)
        label_tensor = torch.tensor(label).long()

        merged = torch.cat((image_tensor, uncertainty_tensor), dim=0)  # [2,D,H,W]

        if self.transform:
            data = self.transform({
                "merged": merged,
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
    probs = torch.sigmoid(logits)

    return (probs > 0.5).sum(dim=1)

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
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False,pin_memory=True,
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
    criterion = coral_loss_manual

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
    for epoch in range(100):
        print(f"Epoch {epoch + 1}/{85}")
        running_loss, correct, total = 0.0, 0, 0

        model.train()

        for data,label, _ in train_loader:

            data, label = data.to(device),label.to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                preds = model(data)  # shape: [B, 3]
                targets = encode_ordinal_targets(label).to(preds.device)
                loss = criterion(preds, targets)


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * data.size(0)
            total += label.size(0)

            with torch.no_grad():
                decoded_preds = decode_predictions(preds)
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
                targets = encode_ordinal_targets(label).to(preds.device)

                loss = criterion(preds, targets)
                val_running_loss += loss.item() * data.size(0)

                with torch.no_grad():
                    decoded_preds = decode_predictions(preds)
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
            labels_idx = [0, 1, 2, 3]
            #best_kappa_cm = confusion_matrix(val_labels_np, val_preds_np, labels=labels_idx)
            best_kappa_preds = val_preds_np.copy()
            best_kappa_labels = val_labels_np.copy()
            best_kappa_report = report
            best_kappa_epoch = epoch
        # Early stopping check

        if epoch < warmup_epochs:
            warmup_scheduler.step()
            print(f"[Warmup] Epoch {epoch + 1}: LR = {optimizer.param_groups[0]['lr']:.6f}")
        else:
            scheduler.step(epoch_val_loss)
            print(f"[Plateau] Epoch {epoch + 1}: LR = {optimizer.param_groups[0]['lr']:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_report = classification_report(val_labels_np, val_preds_np, target_names=class_names, digits=4,
                                           zero_division=0)
            # Save best model weights
            # torch.save({
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'epoch': epoch,
            #     'val_loss': epoch_val_loss
            # }, os.path.join(plot_dir,f"best_qa_model_fold{fold}_early_fusion.pt"))

            np.savez(os.path.join(plot_dir, f"final_preds_fold{fold}_early_fusion_{uncertainty_metric}.npz"), preds=val_preds_np, labels=val_labels_np)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Plot and save best confusion matrix
    # plt.figure(figsize=(6, 5))
    # sns.heatmap(best_kappa_cm, annot=True, fmt='d', cmap='Blues',
    #             xticklabels=class_names, yticklabels=class_names)
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.title(f"Best Confusion Matrix (Epoch {best_kappa_epoch}, κ² = {best_kappa:.3f})")
    # plt.tight_layout()
    # plt.savefig(os.path.join(plot_dir, f"best_conf_matrix_fold{fold}_{uncertainty_metric}_EF.png"))
    # plt.close()

    return train_losses, val_losses, val_preds_list, val_labels_list, val_subtypes_list, f1_history, best_report


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

def main(data_dir, plot_dir, folds,df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # for fold in range(5):
    # Train the model and retrieve losses + predictions

    metrics = ['confidence', 'entropy', 'mutual_info', 'epkl']
    for idx, metric in enumerate(metrics):
        print(f'Training with metric {metric}')
        start = time.time()
        train_losses, val_losses, val_preds, val_labels, val_subtypes, f1_history, best_report = train_one_fold(
            fold=0,
            data_dir=data_dir,
            df=df,
            splits=folds,
            uncertainty_metric= metric,
            plot_dir=plot_dir,
            device=device
        )

        end = time.time()
        print(f'Best Report with metric {metric}: ')
        print(best_report)
        file = os.path.join(plot_dir, f'best_report_{metric}_EF')
        with open(file, "w") as f:
            f.write(f"Final Classification Report for Fold 0:\n")
            f.write(best_report)


        print(f"Total training time: {(end - start) / 60:.2f} minutes")

        # Convert prediction outputs to numpy arrays for plotting
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_subtypes = np.array(val_subtypes)

        plt.figure(figsize=(10, 6))
        for class_name, f1_scores in f1_history.items():
            plt.plot(f1_scores, label=class_name)

        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.title("Per-Class F1 Score Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"f1_scores_per_class_{metric}_EF.png"), dpi=300)
        plt.show()

        # ✅ Plot loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', marker='o')
        plt.plot(val_losses, label='Validation Loss', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss Curves')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'loss_curves_{metric}_EF.png'))
        plt.close()

        # ✅ Overall scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(val_labels, val_preds, c='blue', alpha=0.6, label='Predicted vs Actual')
        plt.plot([val_labels.min(), val_labels.max()],
                 [val_labels.min(), val_labels.max()], 'r--', label='45-degree line')
        plt.xlabel("Actual Label")
        plt.ylabel("Predicted Label")
        plt.title("Overall Predicted vs. Actual Labels")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'pred_vs_actual_{metric}_EF.png'))
        plt.close()


#metrics: confidence, entropy,mutual_info,epkl

if __name__ == '__main__':

    with open('/gpfs/home6/palfken/masters_thesis/Final_splits.json', 'r') as f:
        splits = json.load(f)
    clinical_data = "/gpfs/home6/palfken/masters_thesis/Final_dice_dist1.csv"
    df =  pd.read_csv(clinical_data)

    preprocessed= sys.argv[1]
    plot_dir = sys.argv[2]

    main(preprocessed, plot_dir, splits, df)
