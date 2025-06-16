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

class Extract_ROI(nn.Module):
    def __init__(self, features,mask):
        super(Extract_ROI, self).__init__()
    def reconstruct(self, x):
        pass

class Classifier_Head(nn.Module):
    def __init__(self, roi_features):
        super(Classifier_Head, self).__init__()

    def forward(self, x):


class Subtype_Classifier(nn.Module):
    def __init__(self, encoder, features, masks):
        super().__init__()
        self.encoder = encoder
        encoder_out_channels = encoder.output_channels[-1]

        self.get_roi_features = Extract_ROI(features,masks)

