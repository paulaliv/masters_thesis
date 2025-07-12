import os
import numpy
import sys
from typing import Tuple, Optional
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from scipy.ndimage import label, find_objects
import glob




class ROIPreprocessor:
    def __init__(self,
                 roi_context: Tuple[int, int, int] = (3, 10, 10),
                 target_spacing: Tuple[int, int, int] = (3, 1, 1),
                 safe_as_nifti = False,
                 target_shape: Tuple[int, int, int] = (48,272,256),):
        self.roi_context = roi_context
        self.target_shape = target_shape
        self.target_spacing = target_spacing
        self.save_as_nifti = safe_as_nifti

    def load_nifti(self, filepath: str):
        nii = nib.load(filepath)
        data = nii.get_fdata()
        affine = nii.affine
        return data.astype(np.float32), affine

    def save_nifti(self, data: np.ndarray, affine: np.ndarray, out_path: str):
        nib.save(nib.Nifti1Image(data, affine), out_path)

    def sitk_to_affine(self,sitk_image):
        direction = np.array(sitk_image.GetDirection()).reshape(3, 3)
        spacing = np.array(sitk_image.GetSpacing())
        origin = np.array(sitk_image.GetOrigin())

        affine = np.eye(4)
        affine[:3, :3] = direction * spacing[np.newaxis, :]
        affine[:3, 3] = origin
        return affine

    def revert_resampling(self, image: sitk.Image, original_spacing, original_size, is_label=False):
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(original_spacing)
        resample.SetSize(original_size)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(0)

        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkLinear)

        return resample.Execute(image)


    def resample_image(self, image: sitk.Image, is_label=False):
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        new_size = [int(round(osz * ospc / tspc)) for osz, ospc, tspc in
                    zip(original_size, original_spacing, self.target_spacing)]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(self.target_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(0)

        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkLinear)

        return resample.Execute(image)

    def apply_resampling(self, img_path, is_label=False):
        img_sitk = sitk.ReadImage(img_path)
        return self.resample_image(img_sitk, is_label)
    def apply_reverse_resampling(self, img:np.ndarray, original_spacing, original_size, is_label=False):
        img_sitk = sitk.GetImageFromArray(img)
        return self.revert_resampling(img_sitk, original_spacing, original_size,is_label)


    # def normalize(self, image: np.ndarray):
    #     non_zero = image != 0
    #     if np.sum(non_zero) == 0:
    #         return image
    #     image[non_zero] = (image[non_zero] - np.mean(image[non_zero])) / (np.std(image[non_zero]) + 1e-8)
    #     return image

    def get_roi_bbox(self, mask: np.ndarray):
        labeled, _ = label(mask > 0)
        slices = find_objects(labeled)
        if not slices:
            return (slice(0, mask.shape[0]), slice(0, mask.shape[1]), slice(0, mask.shape[2]))

        zmin, zmax = slices[0][0].start, slices[0][0].stop
        ymin, ymax = slices[0][1].start, slices[0][1].stop
        xmin, xmax = slices[0][2].start, slices[0][2].stop

        zmin = max(0, zmin - self.roi_context[0])
        ymin = max(0, ymin - self.roi_context[1])
        xmin = max(0, xmin - self.roi_context[2])

        zmax = min(mask.shape[0], zmax + self.roi_context[0])
        ymax = min(mask.shape[1], ymax + self.roi_context[1])
        xmax = min(mask.shape[2], xmax + self.roi_context[2])

        return (slice(zmin, zmax), slice(ymin, ymax), slice(xmin, xmax))

    def crop_to_roi(self,image, mask, bbox: Tuple[slice, slice, slice]):
        return image[bbox], mask [bbox]

    def adjust_to_shape(self, img, mask, shape):
        pad_width = []
        slices = []

        for i in range(3):
            diff = shape[i] - img.shape[i]

            if diff > 0:
                # Padding needed
                pad_before = diff // 2
                pad_after = diff - pad_before
                pad_width.append((pad_before, pad_after))
                slices.append(slice(0, img.shape[i]))
            elif diff < 0:
                print(f'ROI is larger that target shape, cropping')
                print(f'Tumor region: {np.sum(mask > 0)}')
                # Cropping needed
                crop_before = (-diff) // 2
                crop_after = (-diff) - crop_before
                pad_width.append((0, 0))
                slices.append(slice(crop_before, img.shape[i] - crop_after))
            else:
                pad_width.append((0, 0))
                slices.append(slice(0, img.shape[i]))

        # Crop if needed
        img = img[slices[0], slices[1], slices[2]]
        mask = mask[slices[0], slices[1], slices[2]]
        print(f'Tumor region after cropping: {np.sum(mask > 0)}')

        return img, mask

    def normalize(self, img):

        nonzero = img[img > 0]
        if nonzero.size > 0:
            mean, std = nonzero.mean(), nonzero.std()
            img = (img - mean) / std
            img[img == -mean / std] = 0  # Ensure padding stays at 0
        return img

    def preprocess_case(self, img_path, mask_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        case_id = os.path.basename(img_path).replace('.nii.gz', '')

        resampled_img_sitk = self.apply_resampling(img_path, is_label=False)
        resampled_mask_sitk = self.apply_resampling(mask_path, is_label=True)

        orig_img = sitk.ReadImage(img_path)
        original_affine = self.sitk_to_affine(orig_img)
        orig_mask = sitk.ReadImage(mask_path)
        original_spacing = orig_img.GetSpacing()
        original_size = orig_img.GetSize()
        #orig_img = sitk.GetArrayFromImage(orig_sitk)
        orig_mask = sitk.GetArrayFromImage(orig_mask)


        # Get bounding box in original mask
        slices_orig = self.get_roi_bbox(orig_mask)  # same as your get_roi_bbox function

        resampled_img = sitk.GetArrayFromImage(resampled_img_sitk)
        resampled_mask = sitk.GetArrayFromImage(resampled_mask_sitk)

        slices = self.get_roi_bbox(resampled_mask)
        cropped_img, cropped_mask = self.crop_to_roi(resampled_img, resampled_mask, slices)
        img_pp = self.normalize(cropped_img)
        resized_img, resized_mask = self.adjust_to_shape(img_pp, cropped_mask, self.target_shape)


        if self.save_as_nifti:
            full_size_img = np.zeros(original_size[::-1], dtype=np.float32)
            full_size_mask = np.zeros(original_size[::-1], dtype=np.uint8) # Note: z, y, x ordering
            # Paste reverted crop into its position
            z1, z2 = slices_orig[0].start, slices_orig[0].stop
            y1, y2 = slices_orig[1].start, slices_orig[1].stop
            x1, x2 = slices_orig[2].start, slices_orig[2].stop


            reverted_crop_img = self.apply_reverse_resampling(resized_img, original_spacing, original_size, is_label=False)
            reverted_crop_mask = self.apply_reverse_resampling(resized_mask, original_spacing,original_size, is_label = True)

            assert sitk.GetArrayFromImage(reverted_crop_img).shape == (
            z2 - z1, y2 - y1, x2 - x1), f"Shape mismatch in image: {sitk.GetArrayFromImage(reverted_crop_img).shape}"
            assert sitk.GetArrayFromImage(reverted_crop_mask).shape == (
            z2 - z1, y2 - y1, x2 - x1), f"Shape mismatch in mask: {sitk.GetArrayFromImage(reverted_crop_mask).shape}"

            full_size_img[z1:z2, y1:y2, x1:x2] = reverted_crop_img
            full_size_mask[z1:z2, y1:y2, x1:x2] = reverted_crop_mask

            affine = original_affine  # Neutral affine, as physical space is lost in cropping and resizing
            self.save_nifti(full_size_img.astype(np.float32), affine, os.path.join(output_dir, f"{case_id}_CROPPED_img.nii.gz"))
            self.save_nifti(full_size_mask.astype(np.uint8), affine, os.path.join(output_dir, f"{case_id}_CROPPED_mask.nii.gz"))
        else:
            np.save( os.path.join(output_dir, f"{case_id}_img.npy"), resized_img.astype(np.float32))
            np.save( os.path.join(output_dir, f"{case_id}_mask.npy"), resized_mask.astype(np.uint8))

        print(f'Processed {case_id}')

    def preprocess_folder(self, image_dir, mask_dir, output_dir):
        image_paths = sorted(glob.glob(os.path.join(image_dir, '*_0000.nii.gz')))

        for img_path in image_paths:
            case_id = os.path.basename(img_path).replace('_0000.nii.gz', '')
            mask_path = os.path.join(mask_dir, f"{case_id}.nii.gz")
            if os.path.exists(mask_path):
                self.preprocess_case(img_path, mask_path, output_dir)

    def preprocess_uncertainty_map(self, umap_path, mask_path, output_path):
        umap_sitk = self.apply_resampling(umap_path, is_label=False)
        mask_sitk = self.apply_resampling(mask_path, is_label=True)

        umap = sitk.GetArrayFromImage(umap_sitk)
        mask = sitk.GetArrayFromImage(mask_sitk)

        slices = self.get_roi_bbox(mask)
        cropped_umap, _= self.crop_to_roi(umap, mask, slices)
        umap_pp = self.adjust_to_shape(cropped_umap, self.target_shape)

        affine = np.eye(4)
        os.makedirs(output_path, exist_ok=True)
        case_id = os.path.basename(umap_path).replace('.nii.gz', '')
        self.save_nifti(umap_pp.astype(np.float32), affine, os.path.join(output_path, f"{case_id}_umap.nii.gz"))


def main():
    input_folder_img = "/gpfs/home6/palfken/nnUNetFrame/nnUNet_raw/Dataset002_SoftTissue/imagesTr/"
    input_folder_mask ="/gpfs/home6/palfken/nnUNetFrame/nnUNet_raw/Dataset002_SoftTissue/labelsTr/"
    output_folder = "/gpfs/home6/palfken/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/COMPLETE_nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/Cropped_nifti/"

    preprocessor = ROIPreprocessor(safe_as_nifti=True)
    preprocessor.preprocess_folder(input_folder_img, input_folder_mask, output_folder)

if __name__ == '__main__':
    main()

