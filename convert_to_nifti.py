import sys
import numpy as np
import nibabel as nib
import os
def main(data_dir,output_dir):
    # Load the .npz file
    #data_dir = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/FeaturesTr'
    image_name = 'DES_0001_cropped_mask.npz'
    feat_name = 'DES_0001_features_roi.npz'
    mask_dir = os.path.join(data_dir, image_name)
    feat_dir = os.path.join(data_dir, feat_name)
    print('Loading images')
    feat = np.load(feat_dir)
    mask = np.load(mask_dir)
    mask_array = mask['arr_0.npy']  # or use the specific key if you saved it with one
    feat_array = feat['arr_0.npy']
    # # Optional: squeeze or select channel
    # if mask_array.ndim == 4:
    #     mask_array = mask_array[0]  # or array = array.squeeze() if appropriate

    # Load reference NIfTI to copy affine/header
    identity_affine = np.eye(4)

    # Save as NIfTI
    print('Saving images')
    nifti_img = nib.Nifti1Image(mask_array.astype(np.float32), identity_affine)
    nib.save(nifti_img, os.path.join(output_dir,'DES_0001_test_mask.nii.gz'))

    nifti_feat = nib.Nifti1Image(feat_array.astype(np.float32), identity_affine)
    nib.save(nifti_feat, os.path_join(output_dir,'DES_0001_test_feat.nii.gz'))

if __name__ == '__main__':
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]

    main(data_dir, output_dir)