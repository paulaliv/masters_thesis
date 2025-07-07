from collections import defaultdict

from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from torch.cpu.amp import autocast

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

import pandas as pd
import numpy as np
import os
import torch.nn as nn
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F
import time
import sys

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
    Compose, LoadImaged, EnsureChannelFirstd, RandFlipd, RandRotate90d, RandGaussianNoised, NormalizeIntensityd,
    ToTensord
)
train_transforms = Compose([
    # Don't use LoadImaged since data is already loaded
    EnsureChannelFirstd(keys=["image"],channel_dim=0),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["image"], prob=0.5, max_k=3),
    RandGaussianNoised(keys="image", prob=0.2),
    ToTensord(keys=["image"]),
])

val_transforms = Compose([
    EnsureChannelFirstd(keys=['image'],channel_dim=0),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ToTensord(keys=['image'])
])



# gt_dir = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnUNet_raw/Dataset002_SoftTissue/labelsTr'
# logits_dir = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/Logits'
# image_dir =  r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnUNet_raw/Dataset002_SoftTissue/imagesTr'
# preprocessed_dir = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_preprocessed/Dataset002_SoftTissue'
# tabular_data_dir = r'/home/bmep/plalfken/my-scratch/Downloads/WORC_data/WORC_all_with_nnunet_ids.csv'
# tabular_data = pd.read_csv(tabular_data_dir)
# pred_fold_paths = {
#     'fold_0': '/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_0/validation',
#     'fold_1': '/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_1/validation',
#     'fold_2': '/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_2/validation',
#     'fold_3': '/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_3/validation',
#     'fold_4': '/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_4/validation',
# }


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

def bin_dice_score(dice, num_bins=5):
    bin_edges = np.linspace(0, 1, num_bins + 1)
    label = np.digitize(dice, bin_edges, right=False) - 1
    return min(label, num_bins - 1)

class QADataset(Dataset):
    def __init__(self, fold, preprocessed_dir, logits_dir, fold_paths, uncertainty_metric, num_bins = 5,transform=None):
        """
        fold: str, e.g. 'fold_0'
        preprocessed_dir: base preprocessed path with .npz images
        pred_fold_paths: dict with predicted mask folder paths per fold
        fold_paths: dict with fold folder paths containing Dice scores & case IDs
        """
        self.fold = fold
        self.num_bins = num_bins
        self.preprocessed_dir = preprocessed_dir
        self.logits_dir = logits_dir
        self.fold_dir = fold_paths[fold]
        self.uncertainty_metric = uncertainty_metric


        # Load Dice scores & case IDs from a CSV or JSON
        # Example CSV: case_id,dice

        self.metadata = pd.read_csv(self.fold_dir)

        # List of case_ids
        self.case_ids = self.metadata['case_id'].tolist()
        self.dice_scores = self.metadata['dice'].tolist()
        self.subtypes = self.metadata['subtype'].tolist()
        self.ds = nnUNetDatasetBlosc2(self.preprocessed_dir)

        self.transform = transform

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        dice_score = self.dice_scores[idx]
        subtype = self.subtypes[idx]

        print(f'Extracting image, logits and label for case {case_id}')
        file = f'{case_id}_resized.pt'
        image = torch.load(os.path.join(self.preprocessed_dir, file))

        # Load preprocessed image (.npz)
        #Check if its not case_id_0000
        assert image.ndim == 4 and image.shape[0] == 1, f"Expected shape (1, H, W, D), but got {image.shape}"

        # Should be: <class 'numpy.ndarray'>

        # Load predicted mask (.nii.gz)

        logits_path = os.path.join(self.logits_dir, f"{case_id}_uncertainty_metrics.npz")
        data = np.load(logits_path)  # shape (H, W, D)
        uncertainty = data[self.uncertainty_metric]
        print(f'Uncertainty map shape: {uncertainty.shape}')
        #logits = np.expand_dims(logits, axis=0)  # to shape: (1, H, W, D)
        #print(f'Logits shape after expansion: {logits.shape}') #(B, num_classes, D, H, W)

        # nnU-Net raw images usually have multiple channels; choose accordingly:
        # Here, just take channel 0 for simplicity:
        #input_image = np.stack([image[0], pred_mask], axis=0)  # (2, H, W, D)
        # Map dice score to category
        print(f'Dice score: {dice_score}')
        label = bin_dice_score(dice_score, num_bins=self.num_bins)
        print(f'Gets label {label}')

        image_tensor = torch.from_numpy(image).float()
        uncertainty_tensor = torch.from_numpy(uncertainty).float()
        label_tensor = torch.tensor(label).long()

        if self.transform:
            image_tensor = self.transform(image_tensor)
            uncertainty_tensor = self.transform(uncertainty_tensor)

        # if logits_tensor.ndim == 5:
        #     logits_tensor = logits_tensor.squeeze(0)  # now shape: [C_classes, D, H, W]

        print('Image tensor shape : ', image_tensor.shape)
        print('Logits tensor shape : ', uncertainty_tensor.shape)
        print('Label tensor shape : ', label_tensor.shape)


        assert image_tensor.shape[2:] == uncertainty_tensor.shape[2:], \
            f"Batch and spatial dimensions must match. Got encoder_out: {image_tensor.shape}, logits: {uncertainty_tensor.shape}"

        # x = torch.cat([image_tensor, logits_tensor], dim=0)
        # print(f'Shape after concatenating: {x.shape}')


        return {
            'image': image_tensor,  # shape (C_total, D, H, W)
            'uncertainty':uncertainty_tensor,
            'label': label_tensor,  # scalar tensor
            'subtype': subtype
        }

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
    max_d = max(item['image'].shape[1] for item in batch)
    max_h = max(item['image'].shape[2] for item in batch)
    max_w = max(item['image'].shape[3] for item in batch)
    target_shape = get_padded_shape((max_d, max_h, max_w))

    images = torch.stack([pad_tensor(item['image'], target_shape) for item in batch])
    uncertainties = torch.stack([pad_tensor(item['uncertainty'], target_shape) for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    subtypes = [item['subtype'] for item in batch]

    return images, uncertainties, labels, subtypes


def train_one_fold(fold,preprocessed_dir, logits_dir, fold_paths, device):
    print(f"Training fold {fold} ...")

    # Initialize datasets for this fold
    train_fold_ids = [f"fold_{i}" for i in range(5) if i != fold]  # other folds for training
    val_fold_id = f"fold_{fold}"  # current fold for validation

    # Combine training folds datasets
    train_datasets = []
    for train_fold in train_fold_ids:
        ds = QADataset(
            fold=train_fold,
            preprocessed_dir=preprocessed_dir,
            logits_dir=logits_dir,
            fold_paths=fold_paths,
            transform=train_transforms

        )
        train_datasets.append(ds)
    train_dataset = ConcatDataset(train_datasets)

    val_dataset = QADataset(
        fold=val_fold_id,
        preprocessed_dir=preprocessed_dir,
        logits_dir=logits_dir,
        fold_paths=fold_paths,
        transform=val_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True,pin_memory=True,
    collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False,pin_memory=True,
    collate_fn=pad_collate_fn)



    # Initialize your QA model and optimizer
    print('Initiating Model')
    model = QAModel(num_classes=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Early stopping variables
    #DOUBLE CHECK THIS
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    scaler = GradScaler()
    for epoch in range(20):
        print(f'Epoch {epoch}')
        model.train()
        train_losses = []
        print(f'before train loader')
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

        for image, uncertainty, label, _ in train_loader:
            image, uncertainty, label = image.to(device),uncertainty.to(device), label.to(device)
            # print('After loading input and moving to device')
            # print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            # print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

            optimizer.zero_grad()
            with autocast():
                preds = model(image, uncertainty)  # shape: [B, 3]
                loss = criterion(preds, label)
                print('after computing loss')
                print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
                print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())


        avg_train_loss = sum(train_losses) / len(train_losses)


        # Validation step
        model.eval()
        val_losses = []
        all_subtypes = []
        all_preds = []
        all_labels = []
        torch.cuda.empty_cache()
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

        with torch.no_grad():
            for image, uncertainty, label, subtype in val_loader:
                image, uncertainty, label = image.to(device), uncertainty.to(device), label.to(device)
                preds = model(image, uncertainty)
                #print(f'prediction shape is {preds.shape}, needs to be squeezed if not [Batchsize,3]')
                val_loss = criterion(preds, label)
                val_losses.append(val_loss.item())

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

                # Convert subtype tensor(s) to Python list of strings or ints (depending on how subtype is stored)
                if isinstance(subtype, torch.Tensor):
                    subtype_list = [s.item() for s in subtype]  # convert tensor values to Python ints
                else:
                    subtype_list = subtype  # if it's already a list of strings

                all_subtypes.extend(subtype_list)


        metrics_per_subtype = defaultdict(lambda: {'preds': [], 'labels': []})

        for p, l, s in zip(all_preds, all_labels, all_subtypes):
            metrics_per_subtype[s]['preds'].append(p)
            metrics_per_subtype[s]['labels'].append(l)

        print("Validation loss per subtype:")
        for subtype, vals in metrics_per_subtype.items():
            if len(vals['preds']) == 0:
                continue
            preds_tensor = torch.tensor(np.stack(vals['preds']))  # shape [N, num_classes]
            labels_tensor = torch.tensor(vals['labels']).long()  # shape [N]
            # Compute loss for this subtype batch
            loss_subtype = F.cross_entropy(preds_tensor, labels_tensor)
            print(f"Subtype: {subtype}, Validation CrossEntropyLoss: {loss_subtype.item():.4f}")

        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model weights
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss
            }, f"best_qa_model_fold{fold}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

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

def main(preprocessed_dir, logits_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()
    #for fold in range(5):
    train_one_fold(0, preprocessed_dir, logits_dir, fold_paths=fold_paths, device=device)
    end = time.time()
    print(f"Total training time: {(end - start) / 60:.2f} minutes")



if __name__ == '__main__':
    fold_paths = {
        'fold_0': '/gpfs/home6/palfken/masters_thesis/fold_0',
        'fold_1': '/gpfs/home6/palfken/masters_thesis/fold_1',
        'fold_2': '/gpfs/home6/palfken/masters_thesis/fold_2',
        'fold_3': '/gpfs/home6/palfken/masters_thesis/fold_3',
        'fold_4': '/gpfs/home6/palfken/masters_thesis/fold_4',
    }
    preprocessed= sys.argv[1]
    uncertainty = sys.argv[2]

    main(preprocessed,uncertainty)
