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
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
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


#encoder = ResidualEncoder(3, 5, (32,64,128,256,320,320,320),nn.Conv2d, 3, ((1, 1), 2, (2, 2), (2, 2), (2, 2)), 2, False, nn.BatchNorm2d, None, None, None, nn.ReLU, None, stem_channels=7)

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
