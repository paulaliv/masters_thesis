from collections import defaultdict

from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from torch.cpu.amp import autocast
from sklearn.metrics import classification_report
import collections
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch.nn as nn
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F
import time
import sys
import json

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ConcatDataset

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
    Compose, LoadImaged, EnsureChannelFirstd, RandFlipd,RandAffined, RandGaussianNoised, NormalizeIntensityd,
    ToTensord
)
train_transforms = Compose([
    EnsureChannelFirstd(keys=["image", "uncertainty"], channel_dim=0),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    RandAffined(
        keys=["image", "uncertainty"],  # apply same affine to both
        prob=1.0,
        rotate_range=[np.pi / 9],
        translate_range=[0.1, 0.1],
        scale_range=[0.1, 0.1],
        mode=('bilinear', 'nearest')  # bilinear for image, nearest for uncertainty (categorical or regression)
    ),
    RandFlipd(keys=["image", "uncertainty"], prob=0.5, spatial_axis=1),
    ToTensord(keys=["image", "uncertainty"])
])
val_transforms = Compose([
    EnsureChannelFirstd(keys=["image", "uncertainty"], channel_dim=0),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ToTensord(keys=["image", "uncertainty"])
])



class Light3DEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
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

class OodDetection(nn.Module):
    def __init__(self,is_train = True):
        super().__init__()
        self.train = is_train
    def compute_cluster(self):
        pass

class QAModel(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.encoder_img = Light3DEncoder()
        self.encoder_unc= Light3DEncoder()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)  # Output = predicted Dice class
        )

    def forward(self, image, uncertainty):
        x1 = self.encoder_img(image)
        x2 = self.encoder_unc(uncertainty)
        merged = torch.cat((x1, x2), dim=1) #[B,128]

        return self.fc(merged)

def bin_dice_score(dice):
    bin_edges = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]  # 6 bins
    label = np.digitize(dice, bin_edges, right=False) - 1
    return min(label, len(bin_edges) - 2)  # ensures label is in [0, 5]


class QADataset(Dataset):
    def __init__(self, case_ids, data_dir, df, uncertainty_metric, num_bins = 5,transform=None):
        """
        fold: str, e.g. 'fold_0'
        preprocessed_dir: base preprocessed path with .npz images
        pred_fold_paths: dict with predicted mask folder paths per fold
        fold_paths: dict with fold folder paths containing Dice scores & case IDs
        """
        #self.fold = fold
        self.num_bins = num_bins
        self.data_dir = data_dir
        self.df = df
        self.uncertainty_metric = uncertainty_metric


        # Load Dice scores & case IDs from a CSV or JSON
        # Example CSV: case_id,dice

        # List of case_ids
        self.case_ids = case_ids.tolist()
        self.df = self.df[self.df['case_id'].isin(self.case_ids)]

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

        image = torch.load(os.path.join(self.data_dir, f'{case_id}_img.npy'))
        if image.ndim == 3:
            image_tensor = image.unsqueeze(0)  # Add channel dim

        assert image.ndim == 4 and image.shape[0] == 1, f"Expected shape (1, H, W, D), but got {image.shape}"

        print(f'Image shape {image.shape}')
        uncertainty = torch.load(os.path.join(self.data_dir, f'{case_id}_{self.uncertainty_metric}.npy'))
        print(f'UMAP shape {uncertainty.shape}')

        # Map dice score to category
        print(f'Dice score: {dice_score}')
        label = bin_dice_score(dice_score)
        print(f'Gets label {label}')

        #image_tensor = torch.from_numpy(image).float()

        # print(f'Image shape {image.shape}')
        uncertainty_tensor = torch.from_numpy(uncertainty).float()

        uncertainty_tensor = uncertainty_tensor.unsqueeze(0)  # Add channel dim

        label_tensor = torch.tensor(label).long()

        if self.transform:
            data = self.transform({
                "image": image,
                "uncertainty": uncertainty_tensor
            })
            image = data["image"]
            uncertainty_tensor = data["uncertainty"]

        # if logits_tensor.ndim == 5:
        #     logits_tensor = logits_tensor.squeeze(0)  # now shape: [C_classes, D, H, W]
        #
        # print('Image tensor shape : ', image.shape)
        #
        # print('Uncertainty tensor shape : ', uncertainty_tensor.shape)
        # print('Label tensor shape : ', label_tensor.shape)




        # assert image_tensor.shape[2:] == uncertainty_tensor.shape[2:], \
        #     f"Batch and spatial dimensions must match. Got encoder_out: {image_tensor.shape}, logits: {uncertainty_tensor.shape}"

        # x = torch.cat([image_tensor, logits_tensor], dim=0)
        # print(f'Shape after concatenating: {x.shape}')


        return image, uncertainty_tensor, label_tensor, subtype

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
    uncertainties = []
    labels = []
    subtypes = []

    for image, uncertainty, label, subtype in batch:
        pad_dims = (
            0, max_w - image.shape[3],
            0, max_h - image.shape[2],
            0, max_d - image.shape[1]
        )
        image = torch.nn.functional.pad(image, pad_dims)
        uncertainty = torch.nn.functional.pad(uncertainty, pad_dims)

        images.append(image)
        uncertainties.append(uncertainty)
        labels.append(label)
        subtypes.append(subtype)

    images_batch = torch.stack(images)
    uncertainties_batch = torch.stack(uncertainties)
    labels_batch = torch.stack(labels)  # works even if labels are scalar tensors
    # case_ids usually strings, so keep as list

    return images_batch, uncertainties_batch, labels_batch, subtypes

# def pad_collate_fn(batch):
#     # batch[i] = (image, uncertainty, label, case_id)
#     max_d = max(item[0].shape[1] for item in batch)  # item[0] is image
#     max_h = max(item[0].shape[2] for item in batch)
#     max_w = max(item[0].shape[3] for item in batch)
#
#     padded_batch = []
#     for image, uncertainty, label, case_id in batch:
#         # Pad image
#         pad_dims = (
#             0, max_w - image.shape[3],
#             0, max_h - image.shape[2],
#             0, max_d - image.shape[1]
#         )
#         image = torch.nn.functional.pad(image, pad_dims)
#
#         # Pad uncertainty the same way
#         uncertainty = torch.nn.functional.pad(uncertainty, pad_dims)
#
#         padded_batch.append((image, uncertainty, label, case_id))
#
#     return padded_batch

# def pad_collate_fn(batch):
#     max_d = max(item['image'].shape[1] for item in batch)
#     max_h = max(item['image'].shape[2] for item in batch)
#     max_w = max(item['image'].shape[3] for item in batch)
#     target_shape = get_padded_shape((max_d, max_h, max_w))
#
#     images = torch.stack([pad_tensor(item['image'], target_shape) for item in batch])
#     uncertainties = torch.stack([pad_tensor(item['uncertainty'], target_shape) for item in batch])
#     labels = torch.stack([item['label'] for item in batch])
#     subtypes = [item['subtype'] for item in batch]
#
#     return images, uncertainties, labels, subtypes


def train_one_fold(fold,data_dir, df, splits, num_bins, uncertainty_metric, device):
    print(f"Training fold {fold} ...")

    train_case_ids = splits[fold]["train"]
    val_case_ids = splits[fold]["val"]

    # Create datasets
    train_dataset = QADataset(
        case_ids=train_case_ids,
        data_dir=data_dir,
        df=df,
        uncertainty_metric=uncertainty_metric,
        num_bins=num_bins,
        transform=train_transforms,
    )
    val_dataset = QADataset(
        case_ids=val_case_ids,
        data_dir=data_dir,
        df=df,
        uncertainty_metric=uncertainty_metric,
        transform=val_transforms,
    )

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True,pin_memory=True,
    collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False,pin_memory=True,
    collate_fn=pad_collate_fn)



    # Initialize your QA model and optimizer
    print('Initiating Model')
    model = QAModel(num_classes=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # Counts = {
    #     0: 66,  # Fail (0-0.1)
    #     1: 22,  # Poor (0.1-0.7)
    #     2: 25,  # Moderate (0.7-0.8)
    #     3: 27,  # Good (0.8-0.9)
    #     4: 7  # Very Good (0.9- 0.95)
    # }

    # Step 1: Define class counts
    class_counts = torch.tensor([66, 22, 25, 27, 7], dtype=torch.float)

    # Step 2: Inverse frequency
    weights = 1.0 / torch.sqrt(class_counts)

    # Step 3 (optional but recommended): Normalize weights to sum to 1
    weights = weights / weights.sum()
    weights = weights.to(device)

    # Step 4: Create the weighted loss
    criterion = nn.CrossEntropyLoss(weight=weights)


    # Early stopping variables
    #DOUBLE CHECK THIS
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    scaler = GradScaler()

    train_losses = []
    val_losses = []

    val_preds_list, val_labels_list, val_subtypes_list = [], [], []
    val_per_class_acc = defaultdict(list)
    for epoch in range(30):
        print(f"Epoch {epoch + 1}/{30}")
        running_loss, correct, total = 0.0, 0, 0

        model.train()

        for image, uncertainty, label, _ in train_loader:
            image, uncertainty, label = image.to(device),uncertainty.to(device), label.to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                preds = model(image, uncertainty)  # shape: [B, 3]
                loss = criterion(preds, label)


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * image.size(0)
            _, predicted = torch.max(preds, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)


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
            for image, uncertainty, label, subtype in val_loader:
                image, uncertainty, label = image.to(device), uncertainty.to(device), label.to(device)

                preds = model(image, uncertainty)
                #print(f'prediction shape is {preds.shape}, needs to be squeezed if not [Batchsize,3]')
                loss = criterion(preds, label)
                val_running_loss += loss.item() * image.size(0)


                _, predicted = torch.max(preds, 1)
                val_correct += (predicted == label).sum().item()
                val_total += label.size(0)

                val_preds_list.extend(predicted.cpu().numpy())
                val_labels_list.extend(label.cpu().numpy())

                # Convert subtype to list
                if isinstance(subtype, torch.Tensor):
                    subtype_list = subtype.cpu().numpy().tolist()
                else:
                    subtype_list = subtype

                val_subtypes_list.extend(subtype_list)


        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total

        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        val_losses.append(epoch_val_loss)
        # avg_val_loss = sum(val_losses) / len(val_losses)

        val_preds_np = np.array(val_preds_list)
        val_labels_np = np.array(val_labels_list)

        # Optional: define class names for nicer output
        class_names = ["Fail (0-0.1)", "Poor (0.1-0.7)", "Moderate(0.7-0.8)", "Good (0.8-0.9)", "Very Good (0.9-0.95)", "Excellent(>0.95)"]

        report = classification_report(val_labels_np, val_preds_np, target_names=class_names, digits=4, zero_division=0)
        print("Validation classification report:\n", report)
        print("Predicted label counts:", collections.Counter(val_preds_list))
        print("True label counts:", collections.Counter(val_labels_list))


        #print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # Save best model weights
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': epoch_val_loss
            }, f"best_qa_model_fold{fold}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return train_losses, val_losses, val_preds_list, val_labels_list, val_subtypes_list


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
    start = time.time()
    #for fold in range(5):
    # Train the model and retrieve losses + predictions
    train_losses, val_losses, val_preds, val_labels, val_subtypes = train_one_fold(
        0,
        data_dir,
        df=df,
        splits=folds,
        uncertainty_metric='confidence',
        num_bins=5,
        device=device
    )

    end = time.time()
    print(f"Total training time: {(end - start) / 60:.2f} minutes")

    # Convert prediction outputs to numpy arrays for plotting
    val_preds = np.array(val_preds)
    val_labels = np.array(val_labels)
    val_subtypes = np.array(val_subtypes)

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
    plt.savefig(os.path.join(plot_dir, 'loss_curves.png'))
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
    plt.savefig(os.path.join(plot_dir, 'pred_vs_actual.png'))
    plt.close()

    # ✅ Per-subtype scatter plots
    unique_subtypes = np.unique(val_subtypes)

    for subtype in unique_subtypes:
        mask = val_subtypes == subtype
        preds_sub = val_preds[mask]
        labels_sub = val_labels[mask]

        plt.figure(figsize=(8, 6))
        plt.scatter(labels_sub, preds_sub, c='green', alpha=0.6)
        plt.plot([labels_sub.min(), labels_sub.max()],
                 [labels_sub.min(), labels_sub.max()], 'r--')
        plt.xlabel("Actual Label")
        plt.ylabel("Predicted Label")
        plt.title(f"Predicted vs Actual for Subtype: {subtype}")
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(os.path.join(plot_dir, f'pred_vs_actual_type_{subtype}.png'))
        plt.close()




#metrics: confidence, entropy,mutual_info,epkl

if __name__ == '__main__':
    with open('/gpfs/home6/palfken/QA_5fold_splits.json', 'r') as f:
        splits = json.load(f)
    clinical_data = "/gpfs/home6/palfken/Dice_scores_5epochs.csv"
    df =  pd.read_csv(clinical_data)

    preprocessed= sys.argv[1]
    plot_dir = sys.argv[2]

    main(preprocessed, plot_dir, splits, df)
