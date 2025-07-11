import sys
import numpy as np
import nibabel as nib
import os
import torch




def convert_npz_to_nii(npz_folder, input_nifti_dir, out_folder, overwrite=False):
    # os.makedirs(out_folder, exist_ok=True)
    npz_files = [f for f in os.listdir(npz_folder) if f.endswith('.pt')]

    for file in npz_files:
        base = file.replace('_resized.pt', '')
        image = torch.load(os.path.join(npz_folder, file))
        mask = torch.load(os.path.join(npz_folder, base + '_mask_resized'), weights_only=False)


        # Check mask values
        unique_vals = np.unique(mask)
        if not np.array_equal(unique_vals, [0, 1]):
            print(f"[Warning] Mask in {file} has unexpected values: {unique_vals}")
            print(f"→ Binarizing mask to [0, 1]")
            mask = (mask == 1).astype(np.uint8)
            print(f'Unique values: {np.unique(mask)}')

        # Assuming image is a PyTorch tensor, convert it:
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            print(f'Image shape: {image.shape}')

        if isinstance(mask, torch.Tensor):
            mask = image.cpu().numpy()
            print(f'Mask shape: {mask.shape}')


        # Make sure dtype is compatible
        image = image.astype(np.float32)  # or np.uint8 for masks


        # Save as .nii.gz

        nii_path = os.path.join(input_nifti_dir, base + '_0000.nii.gz')  # adjust if necessary
        #spacing = (3.6, 0.6944, 0.6944)
        spacing =[1,1,1]
        # tensor shape: [C, Z, Y, X] or [Z, Y, X]
        if image.ndim == 4:
            tensor = image[0]  # take first channel
        affine = np.diag(spacing + (1,))
        if hasattr(image, "numpy"):
            nib_img = nib.Nifti1Image(image.numpy().astype(np.float32), affine)
        else:
            nib_img = nib.Nifti1Image(image.astype(np.float32), affine)






        # Create a simple diagonal affine using spacing
        #affine = np.diag(spacing + (1.0,))




        patient_id = os.path.splitext(file)[0]
        image_out = os.path.join(out_folder, f"{patient_id}_ROI_image.nii.gz")
        mask_out = os.path.join(out_folder, f"{patient_id}_ROI_mask.nii.gz")

        if not overwrite and (os.path.exists(image_out) or os.path.exists(mask_out)):
            print(f"Skipping {patient_id} (already exists)")
            continue

        nib.save(nib_img, image_out)
        nib.save(nib.Nifti1Image(mask, affine), mask_out)

        print(f"✔ Converted: {patient_id}")



data_dir = '/gpfs/home6/palfken/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/COMPLETE_nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/Classification_Tr/'
out_dir = '/gpfs/home6/palfken/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/COMPLETE_nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/Cropped_nifti/'
nifti_folder =  '/gpfs/home6/palfken/nnUNetFrame/nnUNet_raw/Dataset002_SoftTissue/imagesTr'
convert_npz_to_nii(data_dir, nifti_folder, out_dir,overwrite=True)