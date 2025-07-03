import pandas as pd
import os

import numpy as np
import torch
import json
import nibabel as nib

import torch.nn.functional as F

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

og_image_dir = "/gpfs/home6/palfken/nnUNetFrame/nnunet_preprocessed/Task002_SoftTissue/nnUNetPlans_3d_fullres/DES_0001.b2nd"
#og_mask_dir =
meta_data = "/gpfs/home6/palfken/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres_v2/Classification_Tr/DES_0001_meta.json"
resized_image_dir = "/gpfs/home6/palfken/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres_v2/Classification_Tr/DES_0001_resized.pt"
resized_mask_dir = "/gpfs/home6/palfken/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres_v2/Classification_Tr/DES_0001_mask_resized"
output_dir = "/gpfs/home6/palfken/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres_v2/test/"

ds = nnUNetDatasetBlosc2(og_image_dir)
fname = 'DES_0001'
#case_id = fname.replace('.b2nd', '')
data, seg, seg_prev, properties = ds.load_case(fname)
print("Data shape:", data.shape)

image = data
mask = seg
image = torch.tensor(np.array(image))
mask = torch.tensor(np.array(mask))

def save_nifti_from_tensor(tensor, path, spacing=(1, 1, 1)):
    # tensor shape: [C, Z, Y, X] or [Z, Y, X]
    if tensor.ndim == 4:
        tensor = tensor[0]  # take first channel
    affine = np.diag(spacing + (1,))
    nib_img = nib.Nifti1Image(tensor.numpy().astype(np.float32), affine)
    nib.save(nib_img, path)

# ---- Load resized ----
resized_image = torch.load(resized_image_dir)  # shape: [1, Z, Y, X]
resized_mask = torch.load(resized_mask_dir)    # same shape

# ---- Save resized ----

save_nifti_from_tensor(resized_image, os.path.join(output_dir,'resized_image.nii.gz'))
save_nifti_from_tensor(resized_mask,  os.path.join(output_dir,'resized_mask.nii.gz'))

# ---- Load original ----
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

ds = nnUNetDatasetBlosc2(og_image_dir)
fname = 'DES_0001'
image, mask, _, _ = ds.load_case(fname)

image = torch.tensor(np.array(image))  # [1, Z, Y, X]
mask  = torch.tensor(np.array(mask))   # same shape

# ---- Save original ----
save_nifti_from_tensor(image, os.path.join(output_dir,'original_image.nii.gz'))
save_nifti_from_tensor(mask,  'original_mask.nii.gz')

