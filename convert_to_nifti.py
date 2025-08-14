import sys
import numpy as np
import nibabel as nib
import os
import torch
import SimpleITK as sitk




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
        spacing =(1,1,1)
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

def convert_dicom():
    input_root = "/home/bmep/plalfken/my-scratch/test_data/"
    output_root = "/home/bmep/plalfken/my-scratch/test_data_nifti/"
    os.makedirs(output_root, exist_ok=True)
    for patient in os.listdir(input_root):
        patient_path = os.path.join(input_root, patient)
        if not os.path.isdir(patient_path):
            continue

        # get all subfolders (series) and sort to pick the first
        series_folders = sorted([os.path.join(patient_path, f)
                                 for f in os.listdir(patient_path)
                                 if os.path.isdir(os.path.join(patient_path, f))])

        if not series_folders:
            print(f"No series found for {patient}, skipping.")
            continue

        first_series = series_folders[0]  # pick first series
        dicom_folder = os.path.join(first_series, "DICOM")
        if not os.path.exists(dicom_folder):
            print(f"No DICOM folder in {first_series}, skipping.")
            continue

        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
        if len(dicom_names) == 0:
            print(f"No DICOMs found in {dicom_folder}, skipping.")
            continue

        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        output_nii = os.path.join(output_root, f"{patient.replace('-', '_')}.nii.gz")
        if os.path.exists(output_nii):
            print(f"Skipping {patient}")
            continue
        sitk.WriteImage(image, output_nii)
        print(f"Converted {dicom_folder} -> {output_nii}")



convert_dicom()