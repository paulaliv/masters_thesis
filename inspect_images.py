import pandas as pd
import os

import numpy as np
import torch
import json
import nibabel as nib

import sys

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2


def save_nifti_from_tensor(tensor, path, spacing=(1, 1, 1)):
    # tensor shape: [C, Z, Y, X] or [Z, Y, X]
    if tensor.ndim == 4:
        tensor = tensor[0]  # take first channel
    affine = np.diag(spacing + (1,))
    nib_img = nib.Nifti1Image(tensor.numpy().astype(np.float32), affine)
    nib.save(nib_img, path)


def main(case_id):
    og_image_dir = "/gpfs/home6/palfken/nnUNetFrame/nnunet_preprocessed/Task002_SoftTissue/nnUNetPlans_3d_fullres/"
    meta_data = os.path.join("/gpfs/home6/palfken/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres_v2/Classification_Tr/",f'{case_id}_meta.json')
    resized_image_dir = os.path.join("/gpfs/home6/palfken/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres_v2/Classification_Tr/",f"{case_id}_resized.pt")
    resized_mask_dir = os.path.join("/gpfs/home6/palfken/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres_v2/Classification_Tr/",f"{case_id}_mask_resized")
    output_dir = "/gpfs/home6/palfken/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres_v2/test/"

    # ---- Load resized ----
    resized_image = torch.load(resized_image_dir)  # shape: [1, Z, Y, X]
    
    resized_mask = torch.load(resized_mask_dir, weights_only=False)

    # ---- Save resized ----

    save_nifti_from_tensor(resized_image, os.path.join(output_dir,'resized_image.nii.gz'))
    save_nifti_from_tensor(resized_mask,  os.path.join(output_dir,'resized_mask.nii.gz'))

    # ---- Load original ----

    ds = nnUNetDatasetBlosc2(og_image_dir)
    print('og image dir,',og_image_dir)
    image, mask, _, _ = ds.load_case(case_id)

    image = torch.tensor(np.array(image))  # [1, Z, Y, X]
    mask  = torch.tensor(np.array(mask))   # same shape

    # ---- Save original ----
    save_nifti_from_tensor(image, os.path.join(output_dir,'original_image.nii.gz'))
    save_nifti_from_tensor(mask,  os.path.join(output_dir,'original_mask.nii.gz'))



if __name__ == '__main__':
    case_id = sys.argv[1]
    main(case_id)

