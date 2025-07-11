import sys
import numpy as np
import nibabel as nib
import os
import torch


def convert_npz_to_nii(npz_folder, overwrite=False):
    # os.makedirs(out_folder, exist_ok=True)
    npz_files = [f for f in os.listdir(npz_folder) if f.endswith('.pt')]

    for file in npz_files:
        base = file.replace('_resized.pt', '')
        image = torch.load(os.path.join(npz_folder, file))
        mask = torch.load(os.path.join(npz_folder, base + '_mask_resized'))


        # Check mask values
        unique_vals = np.unique(mask)
        if not np.array_equal(unique_vals, [0, 1]):
            print(f"[Warning] Mask in {file} has unexpected values: {unique_vals}")
            # print(f"→ Binarizing mask to [0, 1]")
            # mask = (mask == 1).astype(np.uint8)
        #
        #
        # # Save as .nii.gz
        #
        # nii_path = os.path.join(input_nifti_dir, base + '_0000.nii.gz')  # adjust if necessary
        #
        # # Load original image for affine
        # orig_nii = nib.load(nii_path)
        # affine = orig_nii.affine
        #
        #
        # patient_id = os.path.splitext(file)[0]
        # image_out = os.path.join(out_folder, f"{patient_id}_image.nii.gz")
        # mask_out = os.path.join(out_folder, f"{patient_id}_mask.nii.gz")
        #
        # if not overwrite and (os.path.exists(image_out) or os.path.exists(mask_out)):
        #     print(f"Skipping {patient_id} (already exists)")
        #     continue
        #
        # nib.save(nib.Nifti1Image(image, affine), image_out)
        # nib.save(nib.Nifti1Image(mask, affine), mask_out)
        #
        # print(f"✔ Converted: {patient_id}")


# def main(data_dir,output_dir):
#     # Load the .npz file
#     #data_dir = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/FeaturesTr'
#     crop_name = 'DES_0001_cropped_mask.npz'
#     feat_name = 'DES_0001_features_roi.npz'
#     og_mask = 'DES_0001.nii.gz'
#     mask_dir = os.path.join(data_dir, crop_name)
#     og_mask_dir = os.path.join(data_dir, og_mask)
#     feat_dir = os.path.join(data_dir, feat_name)
#     print('Loading images')
#     feat = np.load(feat_dir)
#     mask = np.load(mask_dir)
#
#     mask_array = mask['arr_0.npy']  # or use the specific key if you saved it with one
#     feat_array = feat['arr_0.npy']
#     # # Optional: squeeze or select channel
#     # if mask_array.ndim == 4:
#     #     mask_array = mask_array[0]  # or array = array.squeeze() if appropriate
#     ref_nii = nib.load(og_mask_dir)
#     affine = ref_nii.affine
#
#     # Load reference NIfTI to copy affine/header
#     identity_affine = np.eye(4)
#
#     # Save as NIfTI
#     print('Saving images')
#     nifti_img = nib.Nifti1Image(mask_array.astype(np.float32), affine)
#     nib.save(nifti_img, os.path.join(output_dir,'DES_0001_test_mask.nii.gz'))
#
#     nifti_feat = nib.Nifti1Image(feat_array.astype(np.float32), affine)
#     nib.save(nifti_feat, os.path.join(output_dir,'DES_0001_test_feat.nii.gz'))
#
# if __name__ == '__main__':
#     data_dir = sys.argv[1]
#     output_dir = sys.argv[2]
#
#     main(data_dir, output_dir)

data_dir = '/gpfs/home6/palfken/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/COMPLETE_nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/Classification_Tr/'
convert_npz_to_nii(data_dir, overwrite=True)