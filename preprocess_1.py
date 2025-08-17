import os
import numpy
import sys
from typing import Tuple
import nibabel as nib
import numpy as np

import SimpleITK as sitk
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.ndimage import label, find_objects
import glob
from scipy.ndimage import zoom
import pandas as pd
from radiomics import featureextractor

class ROIPreprocessor:
    def __init__(self,
                 roi_context: Tuple[int, int, int] = (9, 40, 40),
                 target_spacing: Tuple[int, int, int] = (1,1,3),
                 safe_as_nifti = False,
                 save_umaps = False,
                 target_shape: Tuple[int, int, int] = (48,256,256),):
        self.case_id = None
        self.subtype = None
        self.roi_context = roi_context
        self.target_shape = target_shape
        self.target_spacing = target_spacing
        self.save_as_nifti = safe_as_nifti
        self.save_umaps = save_umaps
        self.cropped_cases = []
        self.empty_masks = []
        self.extractor = featureextractor.RadiomicsFeatureExtractor()


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


    def crop_to_roi(self,image, bbox: Tuple[slice, slice, slice]):
        return image[bbox[0], bbox[1], bbox[2]]

    def get_center_crop_bbox(self,image_shape: Tuple[int, int, int]):
        z, y, x = image_shape
        rz, ry, rx = self.target_shape

        # Ensure the crop doesn't exceed image boundaries
        z_start = max((z - rz) // 2, 0)
        y_start = max((y - ry) // 2, 0)
        x_start = max((x - rx) // 2, 0)

        z_end = min(z_start + rz, z)
        y_end = min(y_start + ry, y)
        x_end = min(x_start + rx, x)

        bbox = (slice(z_start, z_end), slice(y_start, y_end), slice(x_start, x_end))
        return bbox




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

    def visualize_img_and_mask(self, img,  mask, output_dir, gt=False, axis=0):
        summed = np.sum(mask == 1, axis=tuple(i for i in range(mask.ndim) if i != axis))
        idx = np.argmax(summed)

        if axis == 0:
            img_slice = img[idx]
            mask_slice = mask[idx]

        elif axis == 1:
            img_slice = img[:, idx, :]
            mask_slice = mask[:, idx, :]

        elif axis == 2:
            img_slice = img[:, :, idx]
            mask_slice = mask[:, :, idx]

        else:
            raise ValueError("Axis must be 0, 1, or 2.")

        # Create 3 subplots: UMAP, Image, Mask
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        # Image only
        axs[0].imshow(img_slice, cmap='gray')
        axs[0].set_title(f'Image Slice {idx}')
        axs[0].axis('off')

        # Mask only
        axs[1].imshow(mask_slice, cmap='Reds')
        axs[1].set_title(f'Mask Slice {idx}')
        axs[1].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save or close figure as per your original logic
        if gt:
            plt.savefig(os.path.join(output_dir, f'IMG_GT_{self.case_id}.png'))
        else:
            plt.savefig(os.path.join(output_dir, f'IMG_PRED_{self.case_id}.png'))
        plt.close()

    def visualize_umap_and_mask(self,umap, mask, img, name, umap_type, output_dir,axis =0):
        summed = np.sum(mask == 1, axis=tuple(i for i in range(mask.ndim) if i != axis))
        idx = np.argmax(summed)
        assert umap.shape == mask.shape, f"Shape mismatch: umap {umap.shape}, mask {mask.shape}"
        if axis == 0:
            img_slice = img[idx]
            mask_slice = mask[idx]
            umap_slice = umap[idx]
        elif axis == 1:
            img_slice = img[:, idx, :]
            mask_slice = mask[:, idx, :]
            umap_slice = umap[:, idx, :]
        elif axis == 2:
            img_slice = img[:, :, idx]
            mask_slice = mask[:, :, idx]
            umap_slice = umap[:, :, idx]
        else:
            raise ValueError("Axis must be 0, 1, or 2.")

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Left: UMAP
        axs[0].imshow(umap_slice, cmap='viridis')
        axs[0].set_title(f'UMAP Slice {idx}')
        axs[0].axis('off')

        # Right: Image + Mask
        axs[1].imshow(img_slice, cmap='gray')
        axs[1].imshow(mask_slice, cmap='Reds', alpha=0.4)
        axs[1].set_title(f'Image + Mask Slice {idx}')
        axs[1].axis('off')

        # Add figure title here
        plt.suptitle(name, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])


        plt.savefig(os.path.join(output_dir, f'dist_map_{self.case_id}.png'))
        plt.close()

    def visualize_full_row(self, img, pred, umap_dict, dice, output_dir, axis=0):
        """
        Visualize one row of 5 plots:
        - img with mask and pred overlayed
        - 4 uncertainty maps side-by-side from umap_dict

        Parameters:
        - img: 3D np array (image)
        - mask: 3D np array (ground truth mask)
        - pred: 3D np array (prediction mask, can be empty)
        - umap_dict: dict of 4 np arrays for uncertainty maps, keys: confidence, entropy, mutual_info, epkl
        - case_id: string for saving
        - output_dir: folder path for saving
        - axis: axis to slice on (default 0)
        """

        # Choose slice index based on mask presence
        summed = np.sum(pred == 1, axis=tuple(i for i in range(pred.ndim) if i != axis))
        idx = np.argmax(summed)

        # Get slices for img, mask, pred, umaps on chosen axis
        def get_slice(arr):
            if axis == 0:
                return arr[idx]
            elif axis == 1:
                return arr[:, idx, :]
            elif axis == 2:
                return arr[:, :, idx]
            else:
                raise ValueError("Axis must be 0, 1, or 2.")

        img_slice = get_slice(img)

        # If pred empty, create empty pred overlay (zeros)
        if pred.sum() == 0:
            pred_slice = np.zeros_like(img_slice)
        else:
            pred_slice = get_slice(pred)

        # Prepare figure with 5 columns
        fig, axs = plt.subplots(1, 5, figsize=(25, 5))

        # 1) Original image with mask and pred overlay
        axs[0].imshow(img_slice, cmap='gray')
        #axs[0].imshow(mask_slice, cmap='Greens', alpha=0.6, label='GT Mask')
        if pred.sum() > 0:
            axs[0].imshow(pred_slice, cmap='Reds', alpha=0.4, label='Prediction')
        axs[0].set_title('Image + Mask + Pred')
        axs[0].axis('off')

        # 2-5) Plot uncertainty maps with same slice & colormap
        umap_titles = ['Confidence', 'Entropy', 'Mutual Info', 'Epkl']
        for i, key in enumerate(['confidence', 'entropy', 'mutual_info', 'epkl']):
            umap_slice = get_slice(umap_dict[key])
            axs[i + 1].imshow(umap_slice, cmap='viridis')
            axs[i + 1].set_title(f'{umap_titles[i]} Map')
            axs[i + 1].axis('off')

        plt.suptitle(f'Subtype: {self.subtype}', fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save figure
        save_path = os.path.join(output_dir, f'{self.case_id}_full_row.png')
        plt.savefig(save_path)
        plt.close()

    def visualize_img_pred_mask(self,img, pred, mask, dice, output_dir, axis=0):
        """
        Visualize original image, prediction mask, and ground truth mask side by side.

        Parameters:
        - img: numpy array, original image volume (3D)
        - pred: numpy array, predicted mask volume (3D)
        - mask: numpy array, ground truth mask volume (3D)
        - case_id: str, identifier for the case (for saving filename)
        - output_dir: str, directory path to save the figure
        - axis: int, axis to slice along (0, 1, or 2)
        """
        # Handle empty prediction
        if pred.sum() == 0:
            print(f"Warning: Prediction is empty for case {self.case_id}, showing blank prediction.")
            pred = np.zeros_like(mask)

        # Find slice with max mask presence
        summed = np.sum(mask == 1, axis=tuple(i for i in range(mask.ndim) if i != axis))
        idx = np.argmax(summed)

        # Extract slices along the chosen axis
        if axis == 0:
            img_slice = img[idx]
            pred_slice = pred[idx]
            mask_slice = mask[idx]
        elif axis == 1:
            img_slice = img[:, idx, :]
            pred_slice = pred[:, idx, :]
            mask_slice = mask[:, idx, :]
        elif axis == 2:
            img_slice = img[:, :, idx]
            pred_slice = pred[:, :, idx]
            mask_slice = mask[:, :, idx]
        else:
            raise ValueError("Axis must be 0, 1, or 2.")

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        axs[0].imshow(img_slice, cmap='gray')
        axs[0].set_title(f'Original Image (slice {idx})')
        axs[0].axis('off')

        axs[1].imshow(pred_slice, cmap='Reds')
        axs[1].set_title(f'Prediction (slice {idx})')
        axs[1].axis('off')

        axs[2].imshow(mask_slice, cmap='Greens')
        axs[2].set_title(f'Ground Truth Mask (slice {idx})')
        axs[2].axis('off')

        plt.suptitle(f'Case: {self.case_id}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        save_path = os.path.join(output_dir, f"{self.case_id}_img_pred_mask_side_by_side.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved visualization for case {self.case_id} at {save_path}")



    def extract_radiomics_features(self, img, mask):
        features = self.extractor.execute(img, mask)
        return features




    def preprocess_case(self, img_path, mask_path, output_dir):
        case_id = os.path.basename(mask_path).replace('.nii.gz', '')
        orig_img = sitk.ReadImage(img_path)
        orig_mask = sitk.ReadImage(mask_path)
        original_spacing = orig_img.GetSpacing()
        original_origin = orig_img.GetOrigin()  # tuple (x, y, z)
        original_direction = np.array(orig_img.GetDirection()).reshape(3, 3)  # ndarray shape (3,3)

        orig_mask.SetOrigin(orig_img.GetOrigin())
        orig_mask.SetSpacing(orig_img.GetSpacing())
        orig_mask.SetDirection(orig_img.GetDirection())

        img_sitk = self.resample_image(orig_img, is_label=False)
        mask_sitk = self.resample_umap(orig_mask, reference=img_sitk,is_label=True)


        resampled_img = sitk.GetArrayFromImage(img_sitk)  # [Z, Y, X]
        resampled_mask = sitk.GetArrayFromImage(mask_sitk)

        orig_img_array = sitk.GetArrayFromImage(orig_img)
        orig_mask_array = sitk.GetArrayFromImage(orig_mask)
        print(f'Original shape :{orig_mask_array.shape}')
        print(f'Image Shape after reshaping to target spacing: {resampled_img.shape}')

        # Get bounding box in original mask
        slices_orig = self.get_roi_bbox(orig_mask_array)  # same as your get_roi_bbox function
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

        cropped_img= self.crop_to_roi(resampled_img,slices)
        cropped_mask = self.crop_to_roi(resampled_mask,  slices)

        if not np.any(cropped_mask):
            print(f"[SKIP] Tumor missing in cropped mask for {self.case_id}")
            return None
        if resampled_mask.sum() < 5:
            print(f"[SKIP] Tumor too sparse in cropped mask for {self.case_id}")
            return None



        #self.visualize_umap_and_mask(cropped_img, cropped_mask, orig_img_array, self.case_id','empty', 'empty')

        cropped_img_sitk = sitk.GetImageFromArray(cropped_img)
        cropped_img_sitk.SetSpacing(img_sitk.GetSpacing())  # very important!
        cropped_mask_sitk = sitk.GetImageFromArray(cropped_mask)
        #cropped_mask_sitk.SetSpacing(mask_sitk.GetSpacing())
        cropped_mask_sitk.CopyInformation(cropped_img_sitk)

        features = self.extract_radiomics_features(cropped_img_sitk, cropped_mask_sitk)


        tumor_size = self.count_tumor_voxels(resampled_mask)
        img_pp = self.normalize(cropped_img)



        resized_img, resized_mask = self.adjust_to_shape(img_pp, cropped_mask, self.target_shape)


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

        #
        # else:
        #     np.save( os.path.join(output_dir, f"{self.case_id}_img.npy"), resized_img.astype(np.float32))
        #     np.save( os.path.join(output_dir, f"{self.case_id}_mask.npy"), resized_mask.astype(np.uint8))

        print(f'Processed {self.case_id}')
        return features

    def compute_dice(self,gt,pred):
        epsilon = 1e-6
        pred = pred.get_fdata().astype(bool)
        gt = gt.get_fdata().astype(bool)

        intersection = np.logical_and(pred, gt).sum()
        return (2. * intersection) / (pred.sum() + gt.sum() + epsilon)

    def center_pad_to_shape(self,volume, target_shape):
        """
        Pad a 3D volume to match target_shape, centering the original content.
        """
        pad_width = []
        for vol_dim, target_dim in zip(volume.shape, target_shape):
            total_pad = target_dim - vol_dim
            if total_pad < 0:
                # If volume is bigger than target, crop instead of pad
                start = (vol_dim - target_dim) // 2
                end = start + target_dim
                # slice along this dimension
                volume = volume[slice(start, end)] if vol_dim > target_dim else volume
                pad_width.append((0, 0))
            else:
                pad_before = total_pad // 2
                pad_after = total_pad - pad_before
                pad_width.append((pad_before, pad_after))
        return np.pad(volume, pad_width, mode='constant')

    def bin_dice_score(self,dice):
        epsilon = 1e-8
        dice = np.asarray(dice)
        dice_adjusted = dice - epsilon  # Shift slightly left
        bin_edges = [0.1, 0.5, 0.7]
        return np.digitize(dice_adjusted, bin_edges, right=True)

    def preprocess_folder(self, image_dir, mask_dir, gt_dir, output_dir, output_dir_visuals):
        #subtypes_csv = "/gpfs/home6/palfken/masters_thesis/all_folds"
        #"/gpfs/home6/palfken/WORC_train.csv"
        subtypes_csv = "/scratch/bmep/plalfken/test_table1.csv"
        subtypes_df = pd.read_csv(subtypes_csv)
        #print(subtypes_df.columns)
        #
        # image_paths = []
        # for idir in image_dirs:
        #     image_paths.extend(glob.glob(os.path.join(idir, '*_0000.nii.gz')))
        # image_paths = sorted(image_paths)

        image_paths = sorted(glob.glob(os.path.join(image_dir, '*_0000.nii.gz')))
        case_stats = []

        if self.save_umaps:
            #save_path = "/gpfs/home6/palfken/radiomics_features.csv"
            save_path = "/scratch/bmep/plalfken/unc_map_features_test_BAD.csv"
            #save_path = "/gpfs/home6/palfken/Dice_scores_BEST.csv"
        else:
            save_path = "/scratch/bmep/plalfken/radiomics_features_test_BAD.csv"


        if os.path.exists(save_path):
            from_scratch = False
            df = pd.read_csv(save_path)
        else:
            from_scratch = True
            df = pd.DataFrame()
        print(f'PRevious number of rows: {len(df)}')
        for img_path in image_paths:
            case_id = os.path.basename(img_path).replace('_0000.nii.gz', '')
            self.case_id = case_id

            mask_path = os.path.join(mask_dir, f"{case_id}.nii.gz")
            #gt_path = os.path.join(gt_dir,f'{case_id}.nii.gz')


            #pred = nib.load(mask_path)
            # gt = nib.load(gt_path)
            # dice = self.compute_dice(gt, pred)
            # print(f'Dice score: {dice}')


            subtype_row = subtypes_df[subtypes_df['new_accnr'] == case_id]
            if not subtype_row.empty:
                tumor_class = subtype_row.iloc[0]['tumor_class']
                if isinstance(tumor_class, float) and np.isnan(tumor_class):
                    tumor_class = 'Unknown'
                    dist = 'Unknown'
                else:
                    tumor_class = str(tumor_class).strip()

                    dist = subtype_row.iloc[0]['dist']
            else:
                tumor_class = 'Unknown'
                dist = 'Unknown'
                print(f'Case id {case_id}: no subtype in csv file!')

            dice = 0
            self.subtype = tumor_class
            if os.path.exists(img_path):
                print(f'Processing {self.case_id}')
                if self.save_umaps:
                    umap_path = os.path.join(mask_dir, f"{case_id}_uncertainty_maps.npz")
                    subject_stats = self.preprocess_uncertainty_map(img_path=img_path,umap_path=umap_path,gt_path=mask_path, mask_path=mask_path,dice_score = dice, output_path=output_dir, output_dir_visuals=output_dir_visuals)

                    if subject_stats is not None:  # Only proceed if preprocessing succeeded
                        new_row = {
                            "case_id": case_id,
                            "tumor_class": tumor_class,
                            "dist": dist,
                            **subject_stats
                        }
                        # Append or save new_row as needed
                    else:
                        print(f"[SKIP] Case {case_id} skipped due to preprocessing issues")
                        continue

                else:
                   features = self.preprocess_case(img_path, mask_path, output_dir)
                   if features is not None:
                       filtered_features = {k: v for k, v in features.items() if "diagnostics" not in k}

                       new_row = {
                           "case_id": case_id,
                           "tumor_class": tumor_class,
                           **filtered_features}
                   else:
                       print(f"[SKIP] Case {case_id} skipped due to preprocessing issues")
                       continue


                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                print(f'Added {case_id}: {dice}')


                df = df.drop_duplicates(subset='case_id')
                df.to_csv(save_path, index=False)

        print(f'Previous number of rows: {len(df)}')

        # # Load existing results if available
        #
        # if os.path.exists(save_path):
        #     df = pd.read_csv(save_path)
        # else:
        #     df = pd.DataFrame(case_stats)
        #
        # df.drop(df[df.tumor_class == 'Unknown'].index, inplace=True)
        # unique_case_ids = df['case_id'].unique()
        #
        # df_unique = df.drop_duplicates(subset='case_id')
        #
        # df_unique.to_csv(save_path, index=False)
        #
        #
        # print(f'CSV file has {len(df)} rows')
        # #
        # #df_OOD = pd.read_csv(OOD_path)
        # # df = pd.concat([df_ID, df_OOD], ignore_index=True)
        #
        # # Add dice bins column for all data
        # df['dice_bin'] = self.bin_dice_score(df['dice'])
        #
        # print("\nDice Score Distribution and Uncertainty Stats by Tumor Class:")
        #
        # # Find all uncertainty-related columns
        # unc_cols = [c for c in df.columns if c.endswith('unc')]
        #
        # for tumor_class, group in df.groupby('tumor_class'):
        #     print(f"\nTumor Class: {tumor_class}")
        #
        #
        #     # Dice stats
        #     mean_dice = group['dice'].mean()
        #     std_dice = group['dice'].std()
        #     print(f"Mean Dice: {mean_dice:.3f}")
        #     print(f"Std Dice: {std_dice:.3f}")
        #
        #     # Uncertainty stats
        #     if unc_cols:
        #         print("Uncertainty statistics (mean ± std across subjects):")
        #         for col in unc_cols:
        #             col_mean = group[col].mean()
        #             col_std = group[col].std()
        #             print(f"  {col}: {col_mean:.3f} ± {col_std:.3f}")
        #
        #         # Now print uncertainty stats by dice bin
        # print("\nUncertainty stats per Dice bin:")
        # for tumor_class, group in df.groupby('tumor_class'):
        #     if tumor_class == "Lipoma":
        #
        #         for bin_id, bin_group in group.groupby('dice_bin'):
        #             # Map bin_id to readable range
        #             if bin_id == 0:
        #                 bin_range = f"<= 0.1"
        #             elif bin_id == 1:
        #                 bin_range = "0.1 < dice <= 0.5"
        #             elif bin_id == 2:
        #                 bin_range = "0.5 < dice <= 0.7"
        #             else:
        #                 bin_range = "> 0.7"
        #
        #             print(f" Dice bin {bin_id} ({bin_range}): {len(bin_group)} samples")
        #             if len(bin_group) > 0 and unc_cols:
        #                 for col in unc_cols:
        #                     col_mean = bin_group[col].mean()
        #                     col_std = bin_group[col].std()
        #                     print(f"  {col}: {col_mean:.3f} ± {col_std:.3f}")
        #             else:
        #                 print("  No samples in this bin.")



        #df.to_csv("/gpfs/home6/palfken/radiomics_features.csv", index=False)

    def preprocess_uncertainty_map(self, img_path, umap_path, gt_path, mask_path, dice_score, output_path, output_dir_visuals):
        empty_flag = 0
        case_id = os.path.basename(mask_path).replace('.nii.gz', '')
        orig_img = sitk.ReadImage(img_path)
        #orig_mask = sitk.ReadImage(gt_path)
        pred = sitk.ReadImage(mask_path)

        original_spacing = orig_img.GetSpacing()
        original_origin = orig_img.GetOrigin()  # tuple (x, y, z)
        original_direction = np.array(orig_img.GetDirection()).reshape(3, 3)  # ndarray shape (3,3)

        # Convert NumPy array to SimpleITK image

        pred.SetOrigin(orig_img.GetOrigin())
        pred.SetSpacing(orig_img.GetSpacing())
        pred.SetDirection(orig_img.GetDirection())

        img_sitk = self.resample_image(orig_img, is_label=False)
        #orig_mask_sitk = self.resample_umap(orig_mask, reference=img_sitk,is_label=True)
        mask_sitk = self.resample_umap(pred, reference=img_sitk,is_label=True)


        resampled_img = sitk.GetArrayFromImage(img_sitk)  # [Z, Y, X]
        #resampled_mask = sitk.GetArrayFromImage(orig_mask_sitk)
        resampled_pred = sitk.GetArrayFromImage(mask_sitk)

        orig_img_array = sitk.GetArrayFromImage(orig_img)
        #orig_mask_array = sitk.GetArrayFromImage(orig_mask)
        orig_pred_array = sitk.GetArrayFromImage(pred)



        if orig_img_array.shape != orig_pred_array.shape:
            print(f"Shape mismatch: img {orig_img_array.shape}, mask {orig_pred_array.shape}, case {self.case_id}")
            return None  #

        #print(f'Original shape :{orig_mask_array.shape}')
        print(f'Pred shape :{orig_pred_array.shape}')
        print(f'Image Shape after reshaping to target spacing: {resampled_img.shape}')

        # Get bounding box in original mask
        #slices_orig = self.get_roi_bbox(orig_mask_array)  # same as your get_roi_bbox function

        #crop_start_index = np.array([slices_orig[2].start, slices_orig[1].start, slices_orig[0].start])
        #
        # resampled_affine = self.compute_affine_with_origin_shift(
        #     original_spacing, original_origin, original_direction, crop_start_index
        # )

        if resampled_pred.sum() > 0:

            slices = self.get_roi_bbox(resampled_pred)
            for s in slices:
                print(f"Start: {s.start}, Stop: {s.stop}, Length: {s.stop - s.start}")

            cropped_img = self.crop_to_roi(resampled_img,slices)
            cropped_pred = self.crop_to_roi(resampled_pred, slices)

            print("Unique values in resampled_pred:", np.unique(resampled_pred))
            print("Unique values in cropped_pred:", np.unique(cropped_pred))


            cropped_pred_sitk = sitk.GetImageFromArray(cropped_pred)
            cropped_pred_sitk = sitk.GetImageFromArray(cropped_pred.astype(np.uint8))

            cropped_img_sitk = sitk.GetImageFromArray(cropped_img)


            cropped_pred_sitk.CopyInformation(cropped_img_sitk)
            #cropped_pred_sitk.SetSpacing(pred.GetSpacing())


            #cropped_mask = self.crop_to_roi(resampled_mask, slices)

            img_pp = self.normalize(cropped_img)
            resized_img, resized_pred= self.adjust_to_shape(img_pp, cropped_pred, self.target_shape)
            #print('Adjusting GT Mask!!')
            #_,resized_mask = self.adjust_to_shape(img_pp, cropped_mask, self.target_shape)


        else:
            print('WARNING:Prediction is empty, defaulting to center crop')
            self.empty_masks.append(self.case_id)
            img_pp = self.normalize(resampled_img)
            resized_img, resized_pred= self.adjust_to_shape(img_pp, resampled_pred, self.target_shape)


        #self.visualize_umap_and_mask(resized_dist, resized_pred, resized_img, f'{self.case_id}: feature distance map','feature_distance_map', output_dir_visuals)
        stats_dict = {}
        umap_types = ['confidence', 'entropy', 'mutual_info', 'epkl']
        resampled_umaps = {}  # dictionary to store resized UMAPs

        for i, umap_type in enumerate(umap_types):
            npz_file = np.load(umap_path)

            umap_array = npz_file[umap_type]
            umap_array = umap_array.astype(np.float32)  # or whichever key you want
            print(f'UMAP shape : {umap_array.shape}')
            umap_array = np.squeeze(umap_array)
            print(f'UAMP shape after squeeze: {umap_array.shape}')

            if umap_array.ndim == 2:
                print(f"[SKIP] Case {case_id} has only a single slice, shape: {umap_array.shape}")
                return None



            umap_array = self.center_pad_to_shape(umap_array, orig_img_array.shape)

           #self.visualize_umap_and_mask(umap_array, orig_mask_array, orig_img_array, f'{self.case_id}: {umap_type} map', umap_type, output_dir_visuals)

            # Convert NumPy array to SimpleITK image
            orig_umap = sitk.GetImageFromArray(umap_array)
            orig_umap.SetOrigin(orig_img.GetOrigin())
            orig_umap.SetSpacing(orig_img.GetSpacing())
            orig_umap.SetDirection(orig_img.GetDirection())


            umap_sitk = self.resample_umap(orig_umap,reference=img_sitk, is_label=False)

            # assert img_sitk.GetSize() == orig_mask_sitk.GetSize() == umap_sitk.GetSize()

            umap_sitk.SetOrigin(img_sitk.GetOrigin())
            umap_sitk.SetSpacing(img_sitk.GetSpacing())
            umap_sitk.SetDirection(img_sitk.GetDirection())
            resampled_umap = sitk.GetArrayFromImage(umap_sitk)

            if resampled_pred.sum() > 5:
                cropped_umap = self.crop_to_roi(resampled_umap, slices)
                print(f'UMAP shape after crop: {cropped_umap.shape}')
                cropped_umap_sitk = sitk.GetImageFromArray(cropped_umap)
                # new_origin = list(umap_sitk.TransformIndexToPhysicalPoint([slices[2].start,
                #                                                            slices[1].start,
                #                                                            slices[0].start]))
                # cropped_umap_sitk.SetOrigin(new_origin)
                #
                # #cropped_umap_sitk.SetOrigin(umap_sitk.GetOrigin())
                # cropped_umap_sitk.SetSpacing(umap_sitk.GetSpacing())
                #
                # cropped_umap_sitk.SetDirection(umap_sitk.GetDirection())
                cropped_umap_sitk = sitk.GetImageFromArray(cropped_umap.astype(np.float32))
                cropped_umap_sitk.CopyInformation(cropped_pred_sitk)
                #cropped_pred_sitk.CopyInformation(cropped_umap_sitk)


                resized_umap, _ = self.adjust_to_shape(cropped_umap, cropped_pred, self.target_shape)
                print(f'Cropped pred shape: {cropped_pred.shape}')

                # Force casting to Image
                cropped_umap_sitk = sitk.Cast(cropped_umap_sitk, sitk.sitkFloat32)
                cropped_pred_sitk = sitk.Cast(cropped_pred_sitk, sitk.sitkUInt8)


                assert cropped_umap_sitk.GetSize() == cropped_pred_sitk.GetSize()
                assert cropped_umap_sitk.GetSpacing() == cropped_pred_sitk.GetSpacing()
                assert cropped_umap_sitk.GetOrigin() == cropped_pred_sitk.GetOrigin()
                assert cropped_umap_sitk.GetDirection() == cropped_pred_sitk.GetDirection()


                print("Type of cropped_umap_sitk:", type(cropped_umap_sitk))
                print("Type of cropped_pred_sitk:", type(cropped_pred_sitk))
                if isinstance(cropped_pred_sitk, np.ndarray):
                    cropped_pred_sitk = sitk.GetImageFromArray(cropped_pred_sitk)
                    cropped_pred_sitk.CopyInformation(cropped_umap_sitk)

                features = self.extract_radiomics_features(cropped_umap_sitk, cropped_pred_sitk)
                empty_flag = 0


            else:

                resized_umap, _ = self.adjust_to_shape(resampled_umap, resampled_pred, self.target_shape)

                # # Create a "full image" mask (all ones)
                #
                # full_mask = np.ones(resampled_umap.shape, dtype=np.uint8)
                #
                # print(f'fake mask shape: {full_mask.shape}')
                # if full_mask.ndim == 4 and full_mask.shape[0] == 1:
                #     full_mask = full_mask[0]  # squeeze singleton channel dimension
                #
                # # Convert to SimpleITK
                # full_mask_sitk = sitk.GetImageFromArray(full_mask.astype(np.uint8))
                # full_mask_sitk = sitk.Cast(full_mask_sitk, sitk.sitkUInt8)
                # full_mask_sitk.CopyInformation(img_sitk)
                # print("Type of cropped_pred_sitk:", type(full_mask_sitk))
                #
                # # Extract features from entire uncertainty map
                # features = self.extract_radiomics_features(umap_sitk, full_mask_sitk)
                empty_flag = 1
                features = {}



                # Store resized UMAP in the dict for later use
            resampled_umaps[umap_type] = resampled_umap
            pred_mask = (resampled_pred > 0)

            pred_mean_unc = np.mean(resampled_umap[pred_mask]) if pred_mask.sum() > 0 else np.nan

            pred_median_unc = np.median(resampled_umap[pred_mask]) if pred_mask.sum() > 0 else np.nan

            pred_std_unc = np.std(resampled_umap[pred_mask]) if pred_mask.sum() > 0 else np.nan


            # ---- Store in flat dict with prefixed keys ----
            # Merge radiomics features into the stats_dict with a prefix
            for k, v in features.items():
                stats_dict[f"{umap_type}_{k}"] = v

            # Add empty_flag

            stats_dict[f"{umap_type}_pred_mean_unc"] = pred_mean_unc
            stats_dict[f"{umap_type}_pred_median_unc"] = pred_median_unc
            stats_dict[f"{umap_type}_pred_std_unc"] = pred_std_unc


            if self.save_as_nifti:
                reverted_adjusted_umap = self.resample_to_spacing(resized_umap, self.target_spacing, original_spacing,
                                                              is_mask=False)
                #
                # self.save_nifti(reverted_adjusted_umap.astype(np.float32), resampled_affine,
                #                 os.path.join(output_path, f"{self.case_id}_{umap_type}.nii.gz"))
            else:
                np.save(os.path.join(output_path, f"{self.case_id}_{umap_type}.npy"), resized_umap.astype(np.float32))

        if resampled_pred.sum() > 0:
            self.visualize_full_row(resampled_img,resampled_pred,resampled_umaps, dice_score, output_dir_visuals)
            #self.visualize_img_pred_mask(resampled_img,resampled_pred,  dice_score, output_dir_visuals)

        if self.save_as_nifti:

            reverted_adjusted_img = self.resample_to_spacing(resized_img, self.target_spacing, original_spacing,
                                                             is_mask=False)
            reverted_adjusted_mask = self.resample_to_spacing(resized_pred, self.target_spacing, original_spacing,
                                                              is_mask=True)

            # try:
            #     self.save_nifti(reverted_adjusted_img.astype(np.float32), resampled_affine,
            #                     os.path.join(output_path, f"{case_id}_PADDED_img.nii.gz"))
            #     self.save_nifti(reverted_adjusted_mask.astype(np.uint8), resampled_affine,
            #                     os.path.join(output_path, f"{case_id}_PADDED_mask.nii.gz"))
            #     print('Saved Image and mask')
            # except Exception as e:
            #     print(f"Error saving image/mask for case {case_id}: {e}")
        else:
            np.save(os.path.join(output_path, f"{self.case_id}_img.npy"), resized_img.astype(np.float32))
            #np.save(os.path.join(output_path, f"{self.case_id}_mask.npy"), resized_mask.astype(np.uint8))
            np.save(os.path.join(output_path, f"{self.case_id}_pred.npy"), resized_pred.astype(np.uint8))

        stats_dict["empty_bool"] = empty_flag
        print(f'Processed {self.case_id}')
        return stats_dict




def main():

    input_folders_img = ["/gpfs/home6/palfken/QA_imagesTr/","/gpfs/home6/palfken/QA_imagesTs/"]

    input_folders_gt =  ["/gpfs/home6/palfken/QA_labelsTr/","/gpfs/home6/palfken/QA_labelsTs/"]
    #input_folder_img = "/gpfs/home6/palfken/QA_imagesTr/"
    #input_folder_gt = "/gpfs/home6/palfken/QA_imagesTs/"

    #predicted_mask_folder ="/gpfs/home6/palfken/ood_features/id_umaps/"

    #mask_paths = sorted(glob.glob(os.path.join(input_folder_gt, '*.nii.gz')))

    #output_folder_data = "/gpfs/home6/palfken/ood_features/ood_umaps_cropped_30/"
    output_folder_visuals = '/home/bmep/plalfken/my-scratch/test_visuals_BAD'

    #os.makedirs(output_folder_data, exist_ok=True)
    os.makedirs(output_folder_visuals, exist_ok=True)
    # # dice_scores = []
    input_folder_img = sys.argv[1]
    input_folder_gt = sys.argv[2]
    predicted_mask_folder = sys.argv[3]
    output_folder_data = sys.argv[4]





    #preprocessor = ROIPreprocessor(safe_as_nifti=False, save_umaps=True)

    #preprocessor.preprocess_folder(input_folder_img, predicted_mask_folder,input_folder_gt, output_folder_data, output_folder_visuals)


    preprocessor1 = ROIPreprocessor(safe_as_nifti=False, save_umaps=False)

    preprocessor1.preprocess_folder(input_folder_img, predicted_mask_folder,input_folder_gt, output_folder_data, output_folder_visuals)

if __name__ == '__main__':
    main()

