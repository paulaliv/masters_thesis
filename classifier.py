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


from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ConcatDataset

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


class SimpleFeatureClassifier(nn.Module):
    def __init__(self, in_channels=128, num_classes=5):  # e.g., 5 subtypes
        super(SimpleFeatureClassifier, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)  # Reduces to [C, 1, 1, 1]
        self.fc1 = nn.Linear(in_channels, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: [B, C, H, W, D]
        x = self.pool(x).squeeze(-1).squeeze(-1).squeeze(-1)  # [B, C]
        x = self.relu(self.fc1(x))                            # [B, 64]
        x = self.fc2(x)                                       # [B, num_classes]
        return x


