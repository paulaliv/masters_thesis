from collections import defaultdict

from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

from nnunetv2.preprocessing
import pandas as pd
import numpy as np
import os
import torch.nn as nn
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F



from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ConcatDataset

import torch.optim as optim
#metrics:  MAE, MSE, RMSE, Pearson Correlation, Spearman Correlation
#Top-K Error: rank segmentation by quality (for human review)

gt_dir = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnUNet_raw/Dataset002_SoftTissue/labelsTr'
logits_dir = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/Logits'
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



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64, conv_op=nn.Conv3d):
        super().__init__()
        self.block = nn.Sequential(
            conv_op(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            conv_op(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class QAHead(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Output = predicted Dice class
        )

    def forward(self, x):
        x = self.pool(x)
        return self.fc(x)

class QA_Model(nn.Module):
    def __init__(self, encoder, logit_channels, shared_channels = 64):
        super().__init__()
        self.encoder = encoder
        encoder_out_channels = encoder.output_channels[-1]

        self.shared_processor = ConvBlock(
            in_channels=encoder_out_channels + logit_channels,
            out_channels=shared_channels
        )
         # Last stage channels
        self.head = QAHead(input_channels=shared_channels)

    def forward(self, image, logits):
        image_features = self.encoder(image)
        if isinstance(image_features, (list, tuple)):
            #remove skip connections
            image_features = image_features[-1]
        x = torch.cat([image_features, logits], dim=1)  # concat over channels
        shared_features = self.shared_processor(x)
        return self.head(shared_features)



# Instantiate encoder

encoder = ResidualEncoder(
    input_channels=1,
    n_stages=5,
    features_per_stage=[32, 64, 128, 256, 320],
    conv_op=nn.Conv3d,
    kernel_sizes=[3, 3, 3, 3, 3],
    strides=[1, 2, 2, 2, 2],
    n_blocks_per_stage=2,
    conv_bias=False,
    norm_op=nn.InstanceNorm3d,
    norm_op_kwargs={'eps': 1e-5, 'affine': True},
    dropout_op=None,
    dropout_op_kwargs=None,
    nonlin=nn.LeakyReLU,
    nonlin_kwargs={'negative_slope': 1e-2, 'inplace': True},
    block=BasicBlockD,
    return_skips=False
)

# Instantiate model
qa_model = QA_Model(encoder, logit_channels=1)

# Use in forward pass
image = torch.randn(2, 1, 128, 128, 128)
logits = torch.randn(2, 1, 128, 128, 128)

out = qa_model(image, logits)  # shape [2, 3]


class QADataset(Dataset):
    def __init__(self, fold, preprocessed_dir, logits_dir, fold_paths, transform=None):
        """
        fold: str, e.g. 'fold_0'
        preprocessed_dir: base preprocessed path with .npz images
        pred_fold_paths: dict with predicted mask folder paths per fold
        fold_paths: dict with fold folder paths containing Dice scores & case IDs
        """
        self.fold = fold
        self.preprocessed_dir = preprocessed_dir
        self.logits_dir = logits_dir
        self.fold_dir = fold_paths[fold]

        # Load Dice scores & case IDs from a CSV or JSON
        # Example CSV: case_id,dice

        self.metadata = pd.read_csv(self.fold_dir)

        # List of case_ids
        self.case_ids = self.metadata['case_id'].tolist()
        self.dice_scores = self.metadata['dice'].tolist()
        self.subtypes = self.metadata['subtype'].tolist()


        self.transform = transform

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        dice_score = self.dice_scores[idx]
        subtype = self.subtypes[idx]


        # Load preprocessed image (.npz)
        #Check if its not case_id_0000
        b2nd_path = os.path.join(self.preprocessed_dir, f"{case_id}.b2nd")
        data, seg, seg_prev, properties = nnUNetDatasetBlosc2.load_case(b2nd_path)
        print("Data shape:", data.shape)
        image = data[0]

        # Load predicted mask (.nii.gz)
        logits_path = os.path.join(self.logits_dir, f"{case_id}_resampled_logits.npy")
        logits = np.load(logits_path)  # shape (H, W, D)
        logits = np.expand_dims(logits, axis=0)  # to shape: (1, H, W, D)


        # nnU-Net raw images usually have multiple channels; choose accordingly:
        # Here, just take channel 0 for simplicity:
        #input_image = np.stack([image[0], pred_mask], axis=0)  # (2, H, W, D)
        # Map dice score to category
        if dice_score < 0.4:
            label = 0  # low
        elif dice_score < 0.7:
            label = 1  # medium
        else:
            label = 2  # high
        # Convert to torch tensor


        image_tensor = torch.from_numpy(image).float()
        logits_tensor = torch.from_numpy(logits).float()
        label_tensor = torch.tensor(dice_score).long()

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor,logits_tensor, label_tensor, subtype

def train_one_fold(fold,encoder, preprocessed_dir, logits_dir, fold_paths, device):
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
            fold_paths=fold_paths
        )
        train_datasets.append(ds)
    train_dataset = ConcatDataset(train_datasets)

    val_dataset = QADataset(
        fold=val_fold_id,
        preprocessed_dir=preprocessed_dir,
        logits_dir=logits_dir,
        fold_paths=fold_paths
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)



    # Initialize your QA model and optimizer
    model = QA_Model(encoder, logit_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Early stopping variables
    #DOUBLE CHECK THIS
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(100):  # max epochs
        model.train()
        train_losses = []
        for image, logits, label, subtype in train_loader:
            image, logits, label = image.to(device), logits.to(device), label.to(device)
            optimizer.zero_grad()
            preds = model(image, logits)  # shape: [B, 3]
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())


        avg_train_loss = sum(train_losses) / len(train_losses)

        # Validation step
        model.eval()
        val_losses = []
        all_subtypes = []
        all_preds = []
        all_labels = []


        with torch.no_grad():
            for image, logits, label, subtype in val_loader:
                image, logits, label = image.to(device), logits.to(device), label.to(device).long()
                preds = model(image, logits)
                print(f'prediction shape is {preds.shape}, needs to be squeezed if not [Batchsize,3]')
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
            torch.save(model.state_dict(), f"best_qa_model_fold{fold}.pt")
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

