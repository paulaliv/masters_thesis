import os
import numpy
import sys
from typing import Tuple, Optional
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.ndimage import label, find_objects
import glob
from scipy.ndimage import zoom
import pandas as pd



class ROIPreprocessor:
    def __init__(self,
                 roi_context: Tuple[int, int, int] = (3, 10, 10),
                 target_spacing: Tuple[int, int, int] = (1,1,3),
                 safe_as_nifti = False,
                 save_umaps = False,
                 target_shape: Tuple[int, int, int] = (48,256,256),):
        self.case_id = None
        self.roi_context = roi_context
        self.target_shape = target_shape
        self.target_spacing = target_spacing
        self.save_as_nifti = safe_as_nifti
        self.save_umaps = save_umaps
        self.cropped_cases = []


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

    def compute_affine_with_origin_shift(self,original_spacing, original_origin, original_direction, crop_start_index):
        original_spacing = np.array(original_spacing)
        original_origin = np.array(original_origin)
        original_direction = np.array(original_direction).reshape(3, 3)
        crop_start_index = np.array(crop_start_index)

        new_origin = original_origin + original_direction @ (original_spacing * crop_start_index)

        affine = np.eye(4)
        affine[:3, :3] = original_direction * original_spacing[np.newaxis, :]
        affine[:3, 3] = new_origin
        return affine

    def revert_resampling(self, image: sitk.Image, original_spacing, original_size, is_label=False):

        target_shape = np.array(self.target_shape)  # e.g. (38, 272, 256)
        target_spacing = np.array(self.target_spacing)  # e.g. (3.0, 1.0, 1.0)
        original_spacing = np.array(original_spacing)  # e.g. (3.5, 1.2, 0.9)

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

    def resample_to_spacing(self,image, current_spacing, target_spacing, is_mask=False):
        # Calculate zoom factors: (old_spacing / new_spacing)
        zoom_factors = [cs / ts for cs, ts in zip(current_spacing, target_spacing)]

        # For interpolation order:
        # - Use order=1 (linear) for images
        # - Use order=0 (nearest) for masks to preserve label integrity
        order = 0 if is_mask else 1

        # Apply zoom
        resampled = zoom(image, zoom=zoom_factors, order=order)
        return resampled


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

    def resample_umap(self, image: sitk.Image, reference: sitk.Image, is_label=False) -> sitk.Image:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
        resampler.SetTransform(sitk.Transform())
        resampler.SetOutputSpacing(reference.GetSpacing())
        resampler.SetOutputOrigin(reference.GetOrigin())
        resampler.SetOutputDirection(reference.GetDirection())
        resampler.SetSize(reference.GetSize())

        try:
            resampled = resampler.Execute(image)
        except Exception as e:
            print(f"[ERROR] Resampling failed: {str(e)}")
            return None

        return resampled

    def apply_resampling(self, img_sitk, is_label=False):
        #img_sitk = sitk.ReadImage(img_path)
        return self.resample_image(img_sitk, is_label)

    def apply_reverse_resampling(self, img:np.ndarray, original_spacing, original_size, is_label=False):
        img_sitk = sitk.GetImageFromArray(img)
        return self.revert_resampling(img_sitk, original_spacing, original_size,is_label)

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

        bbox = (slice(zmin, zmax), slice(ymin, ymax), slice(xmin, xmax))
        dims = (zmax - zmin, ymax - ymin, xmax - xmin)

        return bbox


    def crop_to_roi(self,image, mask, bbox: Tuple[slice, slice, slice]):
        return image[bbox[0], bbox[1], bbox[2]], mask[bbox[0], bbox[1], bbox[2]]



    def count_tumor_voxels(self, mask):
        tumor_voxels = np.sum(mask > 0)

        # Compute tumor volume in mm³
        voxel_volume = np.prod(self.target_spacing)
        tumor_volume_mm3 = tumor_voxels * voxel_volume
        return tumor_volume_mm3



    def adjust_to_shape(self, img, mask, shape):
        pad_width = []
        slices = []
        print(f'Tumor region: {np.sum(mask > 0)}')

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
                if self.case_id not in self.cropped_cases:
                    self.cropped_cases.append(self.case_id)
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

        # Pad if needed
        if any(p != (0, 0) for p in pad_width):
            img = np.pad(img, pad_width, mode='constant', constant_values=0)
            mask = np.pad(mask, pad_width, mode='constant', constant_values=0)


        tumor_voxels = np.sum(mask > 0)
        print(f'Tumor region after adjustment: {tumor_voxels}')
        if tumor_voxels == 0:
            print('WARNING:Tumor was cropped out of patch! ')


        return img, mask

    def normalize(self, img):

        nonzero = img[img > 0]
        if nonzero.size > 0:
            mean, std = nonzero.mean(), nonzero.std()
            img = (img - mean) / std
            img[img == -mean / std] = 0  # Ensure padding stays at 0
        return img

    def visualize_umap_and_mask(self,umap, mask, title_prefix=''):
        assert umap.shape == mask.shape, f"Shape mismatch: umap {umap.shape}, mask {mask.shape}"

        z_slices = umap.shape[0]

        for i in range(z_slices):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(umap[i], cmap='viridis')
            axes[0].set_title(f'{title_prefix}UMAP Slice {i}')

            axes[1].imshow(mask[i], cmap='gray')
            axes[1].set_title(f'{title_prefix}Mask Slice {i}')

            plt.tight_layout()
            plt.show()

    def preprocess_case(self, img_path, mask_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.case_id = os.path.basename(img_path).replace('.nii.gz', '')

        orig_img = sitk.ReadImage(img_path)
        orig_mask = sitk.ReadImage(mask_path)
        original_spacing = orig_img.GetSpacing()
        original_origin = orig_img.GetOrigin()  # tuple (x, y, z)
        original_direction = np.array(orig_img.GetDirection()).reshape(3, 3)  # ndarray shape (3,3)

        # orig_img = sitk.GetArrayFromImage(orig_img)

        resampled_img_sitk = self.apply_resampling(orig_img, is_label=False)
        resampled_mask_sitk = self.apply_resampling(orig_mask, is_label=True)


        resampled_img = sitk.GetArrayFromImage(resampled_img_sitk)  # [Z, Y, X]
        resampled_mask = sitk.GetArrayFromImage(resampled_mask_sitk)


        orig_mask = sitk.GetArrayFromImage(orig_mask)
        print(f'Original shape :{orig_mask.shape}')
        print(f'Shape after reshaping to target spacing: {resampled_img.shape}')
        # Get bounding box in original mask
        slices_orig = self.get_roi_bbox(orig_mask)  # same as your get_roi_bbox function
        bbox_shape = (
            slices_orig[0].stop - slices_orig[0].start,
            slices_orig[1].stop - slices_orig[1].start,
            slices_orig[2].stop - slices_orig[2].start
        )
        crop_start_index = np.array([slices_orig[2].start, slices_orig[1].start, slices_orig[0].start])

        resampled_affine = self.compute_affine_with_origin_shift(
    original_spacing, original_origin, original_direction, crop_start_index
)

        slices = self.get_roi_bbox(resampled_mask)

        cropped_img, cropped_mask = self.crop_to_roi(resampled_img, resampled_mask, slices)
        print(f'Cropped ROI image shape: {cropped_img.shape}')
        tumor_size= self.count_tumor_voxels(resampled_mask)
        img_pp = self.normalize(cropped_img)
        resized_img, resized_mask = self.adjust_to_shape(img_pp, cropped_mask, self.target_shape)
        print(f'Resized image shape {resized_img.shape}')


        if self.save_as_nifti:

            reverted_adjusted_img = self.resample_to_spacing(resized_img, self.target_spacing, original_spacing, is_mask=False)
            reverted_adjusted_mask = self.resample_to_spacing(resized_mask, self.target_spacing, original_spacing,is_mask=True)

            # affine = original_affine  # Neutral affine, as physical space is lost in cropping and resizing
            # self.save_nifti(full_size_img.astype(np.float32), affine, os.path.join(output_dir, f"{self.case_id}_CROPPED_img.nii.gz"))
            # self.save_nifti(full_size_mask.astype(np.uint8), affine, os.path.join(output_dir, f"{self.case_id}_CROPPED_mask.nii.gz"))

            self.save_nifti(reverted_adjusted_img.astype(np.float32), resampled_affine,
                            os.path.join(output_dir, f"{self.case_id}_PADDED_img.nii.gz"))
            self.save_nifti(reverted_adjusted_mask.astype(np.uint8), resampled_affine,
                            os.path.join(output_dir, f"{self.case_id}_PADDED_mask.nii.gz"))


        else:
            np.save( os.path.join(output_dir, f"{self.case_id}_img.npy"), resized_img.astype(np.float32))
            np.save( os.path.join(output_dir, f"{self.case_id}_mask.npy"), resized_mask.astype(np.uint8))

        print(f'Processed {self.case_id}')

    def compute_dice(self,gt,pred):
        epsilon = 1e-6
        pred = pred.get_fdata().astype(bool)
        gt = gt.get_fdata().astype(bool)

        intersection = np.logical_and(pred, gt).sum()
        return (2. * intersection) / (pred.sum() + gt.sum() + epsilon)


    def preprocess_folder(self, image_dir, mask_dir, gt_dir, output_dir):
        subtypes_csv = "/gpfs/home6/palfken/masters_thesis/all_folds"
        subtypes_df = pd.read_csv(subtypes_csv)


        image_paths = sorted(glob.glob(os.path.join(image_dir, '*_0000.nii.gz')))
        case_stats = []

        for img_path in image_paths:
            case_id = os.path.basename(img_path).replace('_0000.nii.gz', '')
            self.case_id = case_id
            mask_path = os.path.join(mask_dir, f"{case_id}.nii.gz")
            gt_path = os.path.join(gt_dir,f'{case_id}.nii.gz')
            pred = nib.load(mask_path)
            gt = nib.load(gt_path)
            dice = self.compute_dice(gt, pred)
            print(f'Dice score: {dice}')

            if self.save_umaps:
                umap_path = os.path.join(mask_dir,f"{case_id}_uncertainty_maps.npz")

            subtype_row = subtypes_df[subtypes_df['case_id'] == case_id]
            if not subtype_row.empty:
                tumor_class = subtype_row.iloc[0]['subtype']
                tumor_class.strip()
            else:
                tumor_class = 'Unknown'
                print(f'Case id {case_id}: no subtype in csv file!')
            if os.path.exists(mask_path):

                if self.save_umaps:
                    self.preprocess_uncertainty_map(img_path=img_path,umap_path=umap_path,mask_path=gt_path,output_path=output_dir)
                else:
                   self.preprocess_case(img_path, gt_path, output_dir)

                case_stats.append({
                    "case_id": case_id,
                    "tumor_class": tumor_class,
                    "dice_5": dice,

                })
        df = pd.DataFrame(case_stats)
        bin_edges = np.arange(0.0, 1.1, 0.1)

        print("Global Dice Score Distribution:")
        global_hist, _ = np.histogram(df['dice_5'], bins=bin_edges)
        for i in range(len(bin_edges) - 1):
            print(f"{bin_edges[i]:.1f}–{bin_edges[i + 1]:.1f}: {global_hist[i]} samples")

        print("\nDice Score Distribution by Tumor Class:")
        for tumor_class, group in df.groupby('tumor_class'):
            print(f"\nTumor Class: {tumor_class}")
            class_hist, _ = np.histogram(group['dice_5'], bins=bin_edges)
            for i in range(len(bin_edges) - 1):
                print(f"{bin_edges[i]:.1f}–{bin_edges[i + 1]:.1f}: {class_hist[i]} samples")

        # Compute global stats

        print(f'Cases that were cropped: {self.cropped_cases}')
        print(f'Total cropped images: {len(self.cropped_cases)}')

        #df.to_csv("/gpfs/home6/palfken/Dice_scores_5epochs.csv", index=False)

    def preprocess_uncertainty_map(self, img_path, umap_path, mask_path, output_path):

        case_id = os.path.basename(mask_path).replace('.nii.gz', '')
        orig_img = sitk.ReadImage(img_path)
        orig_mask = sitk.ReadImage(mask_path)
        original_spacing = orig_img.GetSpacing()
        original_origin = orig_img.GetOrigin()  # tuple (x, y, z)
        original_direction = np.array(orig_img.GetDirection()).reshape(3, 3)  # ndarray shape (3,3)


        img_sitk = self.apply_resampling(orig_img, is_label=False)
        mask_sitk = self.resample_umap(orig_mask, reference=img_sitk,is_label=True)


        resampled_img = sitk.GetArrayFromImage(img_sitk)  # [Z, Y, X]
        resampled_mask = sitk.GetArrayFromImage(mask_sitk)

        # orig_img = sitk.GetArrayFromImage(orig_sitk)
        orig_mask = sitk.GetArrayFromImage(orig_mask)
        print(f'Original shape :{orig_mask.shape}')
        print(f'Image Shape after reshaping to target spacing: {resampled_img.shape}')

        # Get bounding box in original mask
        slices_orig = self.get_roi_bbox(orig_mask)  # same as your get_roi_bbox function
        bbox_shape = (
            slices_orig[0].stop - slices_orig[0].start,
            slices_orig[1].stop - slices_orig[1].start,
            slices_orig[2].stop - slices_orig[2].start
        )
        crop_start_index = np.array([slices_orig[2].start, slices_orig[1].start, slices_orig[0].start])

        resampled_affine = self.compute_affine_with_origin_shift(
            original_spacing, original_origin, original_direction, crop_start_index
        )

        slices = self.get_roi_bbox(resampled_mask)
        for s in slices:
            print(f"Start: {s.start}, Stop: {s.stop}, Length: {s.stop - s.start}")
        bbox1_shape = (
            slices[0].stop - slices[0].start,
            slices[1].stop - slices[1].start,
            slices[2].stop - slices[2].start
        )

        cropped_img, cropped_mask = self.crop_to_roi(resampled_img, resampled_mask, slices)
        tumor_size = self.count_tumor_voxels(resampled_mask)
        img_pp = self.normalize(cropped_img)

        resized_img, resized_mask = self.adjust_to_shape(img_pp, cropped_mask, self.target_shape)

        umap_types = ['confidence', 'entropy', 'mutual_info', 'epkl']
        for i, umap_type in enumerate(umap_types):
            npz_file = np.load(umap_path)

            umap_array = npz_file[umap_type]
            umap_array = umap_array.astype(np.float32)  # or whichever key you want
            umap_array = np.squeeze(umap_array)
            print("INITIAL UMAP min:", umap_array.min())
            print("INITIAL UMAP max:", umap_array.max())

            print(f'INITIAL UMAP {umap_type} original shape : {umap_array.shape}')
            # Convert NumPy array to SimpleITK image
            orig_umap = sitk.GetImageFromArray(umap_array)


            umap_sitk = self.resample_umap(orig_umap,reference=img_sitk, is_label=False)

            resampled_umap = sitk.GetArrayFromImage(umap_sitk)
            self.visualize_umap_and_mask(resampled_umap, resampled_mask, '')
            print("Resampled mask shape:", resampled_mask.shape)
            print("Resampled UMAP shape:", resampled_umap.shape)
            print("Resampled mask unique:", np.unique(resampled_mask))


            print(f'Resampled {umap_type} stats before crop: min={resampled_umap.min()}, max={resampled_umap.max()}')
            print(f'Bounding box shape {bbox1_shape}')
            z_slice, y_slice, x_slice = bbox1_shape


            print("Mask inside bbox:", resampled_mask[z_slice, y_slice, x_slice].min(), resampled_mask[z_slice, y_slice, x_slice].max())
            print("UMAP inside bbox:", resampled_umap[z_slice, y_slice, x_slice].min(), resampled_umap[z_slice, y_slice, x_slice].max())
            cropped_umap, _ = self.crop_to_roi(resampled_umap, resampled_mask, slices)

            print("Masked UMAP min/max:", cropped_umap.min(), cropped_umap.max())
            print("UMAP min:", cropped_umap.min())
            print("UMAP max:", cropped_umap.max())
            print("UMAP shape:", cropped_umap.shape)
            print("UMAP dtype:", cropped_umap.dtype)

            if not (np.isclose(np.min(cropped_umap), 0.0) and np.isclose(np.max(cropped_umap), 1.0)):
                umap_pp = self.normalize(cropped_umap)
            else:
                umap_pp = cropped_umap

            resized_umap, _ = self.adjust_to_shape(umap_pp, cropped_mask, self.target_shape)
            print(f'Resized UMAP shape {resized_umap.shape}')
            print("UMAP resized min:", resized_umap.min())
            print("UMAP  resized max:", resized_umap.max())
            print("UMAP resized shape:", resized_umap.shape)
            print("UMAP resized dtype:", resized_umap.dtype)

        #     if self.save_as_nifti:
        #         reverted_adjusted_umap = self.resample_to_spacing(resized_umap, self.target_spacing, original_spacing,
        #                                                       is_mask=False)
        #
        #         self.save_nifti(reverted_adjusted_umap.astype(np.float32), resampled_affine,
        #                         os.path.join(output_path, f"{self.case_id}_{umap_type}.nii.gz"))
        #     else:
        #         np.save(os.path.join(output_path, f"{self.case_id}_{umap_type}.npy"), resized_umap.astype(np.float32))
        #
        #
        #
        # if self.save_as_nifti:
        #
        #     reverted_adjusted_img = self.resample_to_spacing(resized_img, self.target_spacing, original_spacing,
        #                                                      is_mask=False)
        #     reverted_adjusted_mask = self.resample_to_spacing(resized_mask, self.target_spacing, original_spacing,
        #                                                       is_mask=True)
        #
        #     try:
        #         self.save_nifti(reverted_adjusted_img.astype(np.float32), resampled_affine,
        #                         os.path.join(output_path, f"{case_id}_PADDED_img.nii.gz"))
        #         self.save_nifti(reverted_adjusted_mask.astype(np.uint8), resampled_affine,
        #                         os.path.join(output_path, f"{case_id}_PADDED_mask.nii.gz"))
        #         print('Saved Image and mask')
        #     except Exception as e:
        #         print(f"Error saving image/mask for case {case_id}: {e}")
        #
        #
        #
        # else:
        #     np.save(os.path.join(output_path, f"{self.case_id}_img.npy"), resized_img.astype(np.float32))
        #     np.save(os.path.join(output_path, f"{self.case_id}_mask.npy"), resized_mask.astype(np.uint8))
        #
        # print(f'Processed {self.case_id}')




def main():
    input_folder_img = "/gpfs/home6/palfken/QA_imagesTs/"
    input_folder_gt ="/gpfs/home6/palfken/QA_labelsTs/"
    predicted_mask_folder = "/gpfs/home6/palfken/QA_imagesTs/output"
    #mask_paths = sorted(glob.glob(os.path.join(input_folder_gt, '*.nii.gz')))

    output_folder = "/gpfs/home6/palfken/Test_umaps/"
    os.makedirs(output_folder, exist_ok=True)
    # dice_scores = []

    preprocessor = ROIPreprocessor(safe_as_nifti=True, save_umaps=True)
    preprocessor.preprocess_folder(input_folder_img, predicted_mask_folder,input_folder_gt, output_folder)

if __name__ == '__main__':
    main()

