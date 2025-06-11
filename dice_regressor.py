from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
import pandas as pd
import numpy as np
import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

#metrics:  MAE, MSE, RMSE, Pearson Correlation, Spearman Correlation
#Top-K Error: rank segmentation by quality (for human review)

gt_dir = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnUNet_raw/Dataset002_SoftTissue/labelsTr'
image_dir =  r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnUNet_raw/Dataset002_SoftTissue/imagesTr'
preprocessed_dir = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_preprocessed/Dataset002_SoftTissue'
tabular_data_dir = r'/home/bmep/plalfken/my-scratch/Downloads/WORC_data/WORC_all_with_nnunet_ids.csv'
tabular_data = pd.read_csv(tabular_data_dir)
pred_fold_paths = {
    'fold_0': '/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_0/validation',
    'fold_1': '/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_1/validation',
    'fold_2': '/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_2/validation',
    'fold_3': '/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_3/validation',
    'fold_4': '/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_4/validation',
}
fold_paths = {
    'fold_0': '/home/bmep/plalfken/my-scratch/nnUNet/fold_0',
    'fold_1': '/home/bmep/plalfken/my-scratch/nnUNet/fold_1',
    'fold_2': '/home/bmep/plalfken/my-scratch/nnUNet/fold_2',
    'fold_3': '/home/bmep/plalfken/my-scratch/nnUNet/fold_3',

    'fold_4': '/home/bmep/plalfken/my-scratch/nnUNet/fold_4',
}

encoder = ResidualEncoder(
    input_channels=2,  # ← use 2 if input = [image, predicted_mask]
    n_stages=5,
    features_per_stage=[32, 64, 128, 256, 320],  # example, match to nnU-Net
    conv_op=nn.Conv3d,  # or nn.Conv2d depending on your data
    kernel_sizes=[3, 3, 3, 3, 3],
    strides=[1, 2, 2, 2, 2],
    n_blocks_per_stage=2,
    conv_bias=False,
    norm_op=nn.InstanceNorm3d,
    norm_op_kwargs={'eps': 1e-5, 'affine': True},
    dropout_op=None,
    nonlin=nn.LeakyReLU,
    nonlin_kwargs={'negative_slope': 1e-2, 'inplace': True},
    block=BasicBlockD,
    return_skips=False
)

class QAHead(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output = predicted Dice score
        )

    def forward(self, x):
        x = self.pool(x)
        return self.fc(x)

class QA_Model(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = QAHead(input_channels=encoder.output_channels[-1])  # Last stage channels

    def forward(self, x):
        features = self.encoder(x)  # Input is [image + pred mask]
        return self.head(features)


class QADataset(Dataset):
    def __init__(self, fold, preprocessed_dir, pred_fold_paths, fold_paths, transform=None):
        """
        fold: str, e.g. 'fold_0'
        preprocessed_dir: base preprocessed path with .npz images
        pred_fold_paths: dict with predicted mask folder paths per fold
        fold_paths: dict with fold folder paths containing Dice scores & case IDs
        """
        self.fold = fold
        self.preprocessed_dir = preprocessed_dir
        self.pred_dir = pred_fold_paths[fold]
        self.fold_dir = fold_paths[fold]

        # Load Dice scores & case IDs from a CSV or JSON
        # Example CSV: case_id,dice
        import pandas as pd
        self.metadata = pd.read_csv(os.path.join(self.fold_dir, 'dice_scores.csv'))

        # List of case_ids
        self.case_ids = self.metadata['case_id'].tolist()
        self.dice_scores = self.metadata['dice'].tolist()

        self.transform = transform

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        dice_score = self.dice_scores[idx]

        # Load preprocessed image (.npz)
        npz_path = os.path.join(self.preprocessed_dir, f"{case_id}.npz")
        image = load_preprocessed_image(npz_path)  # shape (C, H, W, D)

        # Load predicted mask (.nii.gz)
        pred_mask_path = os.path.join(self.pred_dir, f"{case_id}.nii.gz")
        pred_mask = load_pred_mask(pred_mask_path)  # shape (H, W, D)

        # Concatenate image and mask as 2 channels input for QA model
        # image may have multiple channels, use only first channel or mean?
        # nnU-Net raw images usually have multiple channels; choose accordingly:
        # Here, just take channel 0 for simplicity:
        input_image = np.stack([image[0], pred_mask], axis=0)  # (2, H, W, D)

        # Convert to torch tensor
        input_tensor = torch.from_numpy(input_image).float()
        label = torch.tensor(dice_score).float()

        if self.transform:
            input_tensor = self.transform(input_tensor)

        return input_tensor, label


def train_one_fold(fold):
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
            pred_fold_paths=pred_fold_paths,
            fold_paths=fold_paths
        )
        train_datasets.append(ds)

    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset(train_datasets)

    val_dataset = QADataset(
        fold=val_fold_id,
        preprocessed_dir=preprocessed_dir,
        pred_fold_paths=pred_fold_paths,
        fold_paths=fold_paths
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Initialize your QA model and optimizer
    model = QA_Model(encoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()  # or SmoothL1Loss etc. since it's regression

    # Early stopping variables
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(100):  # max epochs
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch).squeeze()
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)

        # Validation step
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_preds = model(x_val).squeeze()
                val_loss = criterion(val_preds, y_val)
                val_losses.append(val_loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model weights
            torch.save(model.state_dict(), f"best_qa_model_fold{fold}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break


for img_fold, path in pred_fold_paths.items():
    for files in os.listdir(path):
        stem = files.replace('.nii.gz', '')



def compute_dice(pred, gt):
    pred = np.asarray(pred == 1, dtype =np.uint8)
    gt= (np.asarray(gt == 1, dtype=np.uint8))

    intersection = np.sum(pred* gt) #true positives
    union = np.sum(pred) + np.sum(gt) #predicted plus ground truth voxels

    dice = 2. * intersection / union if union > 0 else np.nan

    return dice


model = QA_Model(encoder)
