import torch
import torch.nn as nn

from monai.networks.nets import resnet
import torch
from sklearn.metrics import classification_report
import numpy as np
import copy
from monai.metrics import ConfusionMatrixMetric
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, random_split,ConcatDataset

from torch.utils.data import Dataset
import torch.optim as optim
import numpy as np
import pandas as pd
import sys
import os
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

tumor_to_idx = {
    "MyxofibroSarcomas": 0,
    "LeiomyoSarcomas": 1,
    "DTF": 2,
    "MyxoidlipoSarcoma": 3,
    "WDLPS": 4,
}

class TumorClassifier(nn.Module):
    def __init__(self, model_depth=18, in_channels=1, num_classes=5):  # change num_classes to match your setting
        super().__init__()
        # Feature extractor
        self.encoder = resnet.ResNet(
            block_type="BASIC",  # "BASIC" for ResNet18/34, "BOTTLE" for ResNet50+
            layers=[2, 2, 2, 2],  # ResNet18 config
            in_channels=in_channels,
            num_classes=None  # we'll define our own classifier head
        )
        #for feature extraction
        self.pool = nn.AdaptiveAvgPool3d(1)
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        pooled = self.pool(x)
        return self.classifier(pooled)

    def extract_features(self, x):
        x = self.encoder(x)
        pooled = self.pool(x)
        return pooled.view(pooled.size(0), -1) #sahpe (B,512)

class QADataset(Dataset):
    def __init__(self, fold, preprocessed_dir, fold_paths, transform=None):
        """
        fold: str, e.g. 'fold_0'
        preprocessed_dir: base preprocessed path with .npz images
        pred_fold_paths: dict with predicted mask folder paths per fold
        fold_paths: dict with fold folder paths containing Dice scores & case IDs
        """
        self.fold = fold
        self.preprocessed_dir = preprocessed_dir
        self.fold_dir = fold_paths[fold]


        # Load Dice scores & case IDs from a CSV or JSON
        # Example CSV: case_id,dice

        self.metadata = pd.read_csv(self.fold_dir)

        # List of case_ids
        self.case_ids = self.metadata['case_id'].tolist()
        #self.dice_scores = self.metadata['dice'].tolist()
        self.subtypes = self.metadata['subtype'].tolist()
        self.ds = nnUNetDatasetBlosc2(self.preprocessed_dir)

        self.transform = transform

        self.tumor_to_idx = tumor_to_idx

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        #dice_score = self.dice_scores[idx]
        subtype = self.subtypes[idx]

        print(f'Extracting image and label for case {case_id}')



        label_idx = self.tumor_to_idx[subtype]
        # Load preprocessed image (.npz)
        #Check if its not case_id_0000
        data, seg, seg_prev, properties = self.ds.load_case(case_id)
        #print("Data shape:", data.shape)

        image = data
        assert image.ndim == 4 and image.shape[0] == 1, f"Expected shape (1, H, W, D), but got {image.shape}"
        image = np.asarray(image)
        #print(f'Image Shape {image.shape}')
        print(type(image))
        # Should be: <class 'numpy.ndarray'>
        # nnU-Net raw images usually have multiple channels; choose accordingly:
        # Here, just take channel 0 for simplicity:
        #input_image = np.stack([image[0], pred_mask], axis=0)  # (2, H, W, D)
        # Map dice score to category


        image_tensor = torch.from_numpy(image).float()
        label_tensor = torch.tensor(label_idx).long()

        if self.transform:
            image_tensor = self.transform(image_tensor)


        print('Image tensor shape : ', image_tensor.shape)
        print('Label tensor shape : ', label_tensor.shape)


        return {
            'input': image_tensor,  # shape (C_total, D, H, W)
            'label': label_tensor,  # scalar tensor
        }


def train_one_fold(model, preprocessed_dir, fold_paths, criterion, optimizer, scheduler, num_epochs, patience, device,fold):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    patience_counter = 0

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
            fold_paths=fold_paths
        )
        train_datasets.append(ds)
    train_dataset = ConcatDataset(train_datasets)

    val_dataset = QADataset(
        fold=val_fold_id,
        preprocessed_dir=preprocessed_dir,
        fold_paths=fold_paths
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        running_loss, correct, total = 0.0, 0, 0
        preds_list, labels_list = [], []

        for batch in train_loader:
            inputs = batch['input'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())


        epoch_train_loss = running_loss / total
        epoch_train_acc = correct.double() / total
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")

        idx_to_tumor = {v: k for k, v in tumor_to_idx.items()}
        pred_tumors = [idx_to_tumor[p] for p in preds_list]
        true_tumors = [idx_to_tumor[t] for t in labels_list]

        print(classification_report(true_tumors, pred_tumors, digits=4))


        # --- Validation phase ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_preds_list, val_labels_list = [], []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                labels = batch['label'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)

                val_loss += loss.item() * inputs.size(0)
                val_correct += torch.sum(preds == labels.data)
                val_total += labels.size(0)
                val_preds_list.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct.double() / val_total

        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        val_pred_tumors = [idx_to_tumor[p] for p in val_preds_list]
        val_true_tumors = [idx_to_tumor[t] for t in val_labels_list]
        print(classification_report(val_true_tumors, val_pred_tumors, digits=4))

        scheduler.step(epoch_val_loss)

        # Log current learning rate(s)
        for i, param_group in enumerate(optimizer.param_groups):
            print(f"LR after scheduler step (param group {i}): {param_group['lr']:.6f}")


        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, f"best_model_fold_{fold}.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping")
                model.load_state_dict(best_model_wts)
                return model

    model.load_state_dict(best_model_wts)
    return model

def main(preprocessed_dir, fold_paths,device):
    for fold in range(1):
        model = TumorClassifier(...)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
        criterion = nn.CrossEntropyLoss()

        best_model = train_one_fold(model, preprocessed_dir, fold_paths,criterion, optimizer, scheduler,
                                    num_epochs=20, patience=5, device=device, fold=fold)



if __name__ == '__main__':
    fold_paths = {
        'fold_0': '/gpfs/home6/palfken/masters_thesis/fold_0',
        'fold_1': '/gpfs/home6/palfken/masters_thesis/fold_1',
        'fold_2': '/gpfs/home6/palfken/masters_thesis/fold_2',
        'fold_3': '/gpfs/home6/palfken/masters_thesis/fold_3',
        'fold_4': '/gpfs/home6/palfken/masters_thesis/fold_4',
    }
    preprocessed= sys.argv[1]


    main(preprocessed, fold_paths, device = 'cude')

