from collections import defaultdict

from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

import pandas as pd
import numpy as np
import os
import torch.nn as nn
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F
import time
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import classification_report

import torch.optim as optim

# class Extract_ROI(nn.Module):
#     def __init__(self, features,mask):
#         super(Extract_ROI, self).__init__()
#     def reconstruct(self, x):
#         pass
#
# class Classifier_Head(nn.Module):
#     def __init__(self, roi_features):
#         super(Classifier_Head, self).__init__()
#
#     def forward(self, x):
#
#
# class Subtype_Classifier(nn.Module):
#     def __init__(self, encoder, features, masks):
#         super().__init__()
#         self.encoder = encoder
#         encoder_out_channels = encoder.output_channels[-1]
#
#         self.get_roi_features = Extract_ROI(features,masks)


class SimpleClassifier(nn.Module):
    def __init__(self, in_channels=128, num_classes=5):  # e.g., 5 subtypes
        super(SimpleClassifier, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)  # Reduces to [C, 1, 1, 1]
        self.fc1 = nn.Linear(in_channels, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: [B, C, H, W, D]
        x = self.pool(x).squeeze(-1).squeeze(-1).squeeze(-1)  # [B, C]
        x = self.relu(self.fc1(x))
        x = self.dropout # [B, 64]
        x = self.fc2(x)                                       # [B, num_classes]
        return x


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.features)


X_trainval, X_test, y_trainval, y_test = train_test_split(features, labels, test_size=0.15, stratify=labels, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=42)

train_loader = DataLoader(FeatureDataset(X_train, y_train), batch_size=16, shuffle=True)
val_loader = DataLoader(FeatureDataset(X_val, y_val), batch_size=16)
test_loader = DataLoader(FeatureDataset(X_test, y_test), batch_size=16)


def train_model(model, train_loader, val_loader, num_epochs=100, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = np.inf
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                val_loss += criterion(outputs, y).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model
    model.load_state_dict(best_model_state)
    return model



def evaluate(model, loader):
    device = next(model.parameters()).device
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    print(classification_report(all_labels, all_preds, digits=3))
