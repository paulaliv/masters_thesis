import torch
import torch.nn as nn

from monai.networks.nets import ResNet
import torch
from sklearn.metrics import classification_report
import copy
from monai.metrics import ConfusionMatrixMetric
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, random_split,ConcatDataset

from torch.utils.data import Dataset
import torch.optim as optim
import numpy as np
import pandas as pd
import sys
import os
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import umap



os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
        self.encoder = ResNet(
            block="basic",  # "BASIC" for ResNet18/34, "BOTTLE" for ResNet50+
            layers=[2, 2, 2, 2],  # ResNet18 config
            n_input_channels=in_channels,
            num_classes=5,
            feed_forward=False,
            block_inplanes=[64, 128, 256, 512]

            # we'll define our own classifier head
        )
        #for feature extraction
        #self.pool = nn.AdaptiveAvgPool3d(1)
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
        #print(f"x.shape before pooling: {x.shape}")
        #pooled = self.pool(x)
        return self.classifier(x)

    def extract_features(self, x):
        x = self.encoder(x)
        #pooled = self.pool(x)
        return x.view(x.size(0), -1) #sahpe (B,512)

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
        #self.ds = nnUNetDatasetBlosc2(self.preprocessed_dir)

        self.transform = transform

        self.tumor_to_idx = tumor_to_idx

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        #dice_score = self.dice_scores[idx]
        subtype = self.subtypes[idx]
        subtype = subtype.strip()

        #print(f'Extracting image and label for case {case_id}')



        label_idx = self.tumor_to_idx[subtype]
        # Load preprocessed image (.npz)
        #Check if its not case_id_0000
        # data, seg, seg_prev, properties = self.ds.load_case(case_id)
        #print("Data shape:", data.shape)
        file = f'{case_id}_resized.pt'
        image = torch.load(os.path.join(self.preprocessed_dir, file))

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


        # print('Image tensor shape : ', image_tensor.shape)
        # print('Label tensor shape : ', label_tensor.shape)


        return {
            'input': image_tensor,  # shape (C_total, D, H, W)
            'label': label_tensor,  # scalar tensor
        }


def train_one_fold(model, preprocessed_dir, plot_dir, fold_paths, criterion, optimizer, scheduler, num_epochs, patience, device,fold):
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
    del train_datasets

    val_dataset = QADataset(
        fold=val_fold_id,
        preprocessed_dir=preprocessed_dir,
        fold_paths=fold_paths
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, pin_memory=True, num_workers=4)

    train_losses = []  # <-- add here, before the epoch loop
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        print(f"Epoch {epoch+1}/{num_epochs}")
        running_loss, correct, total = 0.0, 0, 0
        preds_list, labels_list = [], []

        scaler = GradScaler()

        for batch in train_loader:
            inputs = batch['input'].to(device)
            labels = batch['label'].to(device)
            #print("Input shape:", inputs.shape)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
                preds_cpu = preds.detach().cpu()
                labels_cpu = labels.detach().cpu()


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

            if epoch % 5 == 0:
                preds_list.extend(preds_cpu.numpy())
                labels_list.extend(labels_cpu.numpy())



        epoch_train_loss = running_loss / total
        epoch_train_acc = correct.double() / total
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")

        train_losses.append(epoch_train_loss)

        if epoch % 5 == 0:
            idx_to_tumor = {v: k for k, v in tumor_to_idx.items()}
            pred_tumors = [idx_to_tumor[p] for p in preds_list]
            true_tumors = [idx_to_tumor[t] for t in labels_list]

            print(classification_report(true_tumors, pred_tumors, digits=4, zero_division=0))
        del inputs, outputs,labels, preds
        torch.cuda.empty_cache()

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
                preds_cpu = preds.detach().cpu()
                labels_cpu = labels.detach().cpu()

                val_loss += loss.item() * inputs.size(0)
                val_correct += torch.sum(preds == labels.data)
                val_total += labels.size(0)
                val_preds_list.extend(preds_cpu.numpy())
                val_labels_list.extend(labels_cpu.numpy())

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct.double() / val_total

        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        val_losses.append(epoch_val_loss)

        val_pred_tumors = [idx_to_tumor[p] for p in val_preds_list]
        val_true_tumors = [idx_to_tumor[t] for t in val_labels_list]
        print(classification_report(val_true_tumors, val_pred_tumors, digits=4, zero_division=0))



        scheduler.step(epoch_val_loss)

        # Log current learning rate(s)
        for i, param_group in enumerate(optimizer.param_groups):
            print(f"LR after scheduler step (param group {i}): {param_group['lr']:.6f}")


        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            best_report = classification_report(val_true_tumors, val_pred_tumors, digits=4, zero_division=0)
            disp = ConfusionMatrixDisplay(confusion_matrix=best_report, display_labels=list(idx_to_tumor.values()))

            print(f"✅ New best model saved at epoch {epoch + 1} with val loss {epoch_val_loss:.4f}")

            torch.save(best_model_wts, f"best_model_fold_{fold}.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping")
                model.load_state_dict(best_model_wts)

                disp.plot(xticks_rotation=45)
                plt.title(f"Confusion Matrix - Fold {fold}")
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"confusion_matrix_fold_{fold}.png"))
                plt.close()
                file = os.path.join(plot_dir, f"classification_report_fold_{fold}.txt")

                print('Best Report')
                print(best_report)

                with open(file, "w") as f:
                    f.write(f"Final Classification Report for Fold {fold}:\n")
                    f.write(best_report)
                return model, train_losses, val_losses


    #model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses

def plot_UMAP(train, y_train, neighbours, m, name, image_dir):

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

    # Map labels back to names (optional)
    idx_to_tumor = {v: k for k, v in tumor_to_idx.items()}
    label_names_val = [idx_to_tumor[i] for i in y_val]

    label_names_train = [idx_to_tumor[i] for i in y_train]

    # combined_umap = np.vstack([train_umap, val_umap])


    # all_subtypes= np.concatenate([y_train, y_val])
    unique_subtypes = sorted(set(y_train))

    # labels = np.array(['train'] * len(train_umap) + ['val'] * len(val_umap))
    markers = {'train': 'o', 'val': 's'}

    color_lookup = {lab: cmap(i % 20) for i, lab in enumerate(unique_subtypes)}
    # 7. scatter plot
    plt.figure(figsize=(8, 6))
    # for marker_type in ['train', 'val']:
    for subtype in unique_subtypes:
        idx = [i for i, lab in y_train if lab == subtype]
        if not idx: continue
        plt.scatter(
            train_umap[idx, 0], train_umap[idx, 1],
            s=25,
            c=[color_lookup[subtype]] * len(idx),
            label=f"{idx_to_tumor[subtype]} ",
            alpha=0.8,
        )

    plt.xlabel("UMAP‑1")
    plt.ylabel("UMAP‑2")
    plt.title("ROI Feature Map Clusters by Subtype and Set")
    plt.legend(fontsize=8, loc='best', markerscale=1)
    plt.tight_layout()
    image_loc = os.path.join(image_dir, name)
    plt.savefig(image_loc, dpi=300)
    plt.show()

def main(preprocessed_dir, plot_dir, fold_paths, device):
    for fold in range(1):
        model = TumorClassifier(...)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
        criterion = nn.CrossEntropyLoss()

        best_model, train_losses, val_losses= train_one_fold(model, preprocessed_dir, plot_dir,fold_paths,criterion, optimizer, scheduler,
                                    num_epochs=100, patience=10, device=device, fold=fold)

        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curves')
        plt.savefig(os.path.join(plot_dir, 'loss_curves.png'))

def extract_features(train_dir,device, plot_dir):
    model = TumorClassifier(model_depth=18, in_channels=1, num_classes=5)
    model.load_state_dict(torch.load("best_model_fold_0.pth", map_location=device))
    model.to(device)
    model.eval()
    train_dataset = QADataset(
        fold='fold_0',
        preprocessed_dir=train_dir,
        fold_paths=fold_paths
    )
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=4)

    # val_dataset = QADataset(
    #     fold='ood_val',
    #     preprocessed_dir=val_dir,
    #     fold_paths=fold_paths
    # )
    # val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

    # all_features_val = []
    # all_labels_val = []

    all_features_train = []
    all_labels_train = []

    # with torch.no_grad():
    #     for batch in val_loader:
    #         inputs = batch['input'].to(device)  # (B, C, D, H, W)
    #         labels = batch['label'].cpu().numpy()  # class indices
    #
    #         features = model.extract_features(inputs).cpu().numpy()  # (B, 512)
    #         all_features_val.append(features)
    #         all_labels_val.extend(labels)
    #
    # # Combine into arrays
    # X_val = np.concatenate(all_features_val, axis=0)
    # y_val = np.array(all_labels_val)

    with torch.no_grad():
        for batch in train_loader:
            inputs = batch['input'].to(device)  # (B, C, D, H, W)
            labels = batch['label'].cpu().numpy()  # class indices

            features = model.extract_features(inputs).cpu().numpy()  # (B, 512)
            all_features_train.append(features)
            all_labels_train.extend(labels)

    # Combine into arrays
    X_train = np.concatenate(all_features_train, axis=0)
    y_train = np.array(all_labels_train)

    plot_UMAP(X_train,y_train,neighbours=10, m='cosine', name='UMAP_cosine_10n_fold0', image_dir =plot_dir)





if __name__ == '__main__':
    fold_paths = {
        'fold_0': '/gpfs/home6/palfken/masters_thesis/fold_0',
        'fold_1': '/gpfs/home6/palfken/masters_thesis/fold_1',
        'fold_2': '/gpfs/home6/palfken/masters_thesis/fold_2',
        'fold_3': '/gpfs/home6/palfken/masters_thesis/fold_3',
        'fold_4': '/gpfs/home6/palfken/masters_thesis/fold_4',
        'ood_val': '/gpfs/home6/palfken/masters_thesis/ood_val',
    }
    preprocessed= sys.argv[1]

    plot_dir = sys.argv[2]


    # main(preprocessed, plot_dir, fold_paths, device = 'cuda')
    extract_features(preprocessed, device = 'cuda', plot_dir = plot_dir)

