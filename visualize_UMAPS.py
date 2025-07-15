import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob
import nibabel as nib

def visualize_umap_gradcam_overlay(image_3d, umap_3d, mask_3d, slice_strategy='middle', alpha=0.5, cmap=cv2.COLORMAP_JET, title=None):
    """
    Visualizes a UMAP (uncertainty map) overlay on a selected 2D image slice using a Grad-CAM style heatmap.

    Parameters:
    - image_3d (np.ndarray): 3D array of the image (Z, Y, X)
    - umap_3d (np.ndarray): 3D uncertainty map array (Z, Y, X), same shape as image_3d
    - mask_3d (np.ndarray, optional): 3D binary mask to locate tumor for slicing
    - slice_strategy (str): 'middle', 'max_tumor', or an integer index
    - alpha (float): Transparency of the overlay (0: only image, 1: only heatmap)
    - cmap (int): OpenCV colormap (e.g., cv2.COLORMAP_JET)
    - title (str): Optional plot title

    Returns:
    - None (displays plot)
    """
    image_3d = nib.load(image_3d)
    umap_3d = nib.load(umap_3d)
    mask_3d = nib.load(mask_3d)
    assert image_3d.shape == umap_3d.shape, "Image and UMAP shapes must match"

    # Determine slice index
    def get_middle_tumor_slice(mask):
        z_indices = np.any(mask, axis=(1, 2))
        tumor_slices = np.where(z_indices)[0]
        if tumor_slices.size == 0:
            return image_3d.shape[0] // 2  # fallback
        return tumor_slices[len(tumor_slices) // 2]

    if isinstance(slice_strategy, int):
        slice_idx = slice_strategy
    elif mask_3d is not None and slice_strategy == 'max_tumor':
        slice_idx = np.argmax([np.sum(slice) for slice in mask_3d])
    elif mask_3d is not None and slice_strategy == 'middle':
        slice_idx = get_middle_tumor_slice(mask_3d)
    else:
        slice_idx = image_3d.shape[0] // 2  # fallback

    # Slice selection
    image_slice = image_3d[slice_idx]
    umap_slice = umap_3d[slice_idx]
    mask_slice = mask_3d[slice_idx] if mask_3d is not None else None

    # Normalize image
    #image_norm = (image_slice - np.min(image_slice)) / (np.max(image_slice) - np.min(image_slice) + 1e-8)
    #image_rgb = np.stack([image_norm]*3, axis=-1)

    # Normalize umap and convert to heatmap
    #umap_norm = (umap_slice - np.min(umap_slice)) / (np.max(umap_slice) - np.min(umap_slice) + 1e-8)
    #umap_uint8 = np.uint8(255 * umap_norm)
    heatmap = cv2.applyColorMap(umap_slice, cmap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = cv2.addWeighted(heatmap, alpha, (image_slice * 255).astype(np.uint8), 1 - alpha, 0)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    if mask_slice is not None:
        plt.contour(mask_slice, levels=[0.5], colors='white', linewidths=1)
    plt.axis('off')
    plt.title(title or f'UMAP Grad-CAM Overlay | Slice {slice_idx}')
    plt.show()


def main():
    # input_folder_img = "/gpfs/home6/palfken/QA_imagesTs/"
    # input_folder_gt = "/gpfs/home6/palfken/QA_labelsTs/"
    # predicted_mask_folder = "/gpfs/home6/palfken/QA_input_Ts/output"

    Umap_folder = "/gpfs/home6/palfken/Test_umaps/"
    mask_paths = sorted(glob(os.path.join(Umap_folder, '*_PADDED_mask.nii.gz')))
    for mask in range(2):
        mask_path = mask_paths[mask]
        if mask_path:
            print(mask_path)
        case_id = os.path.basename(mask_path).replace('_PADDED_mask.nii.gz', '')
        img_path = os.path.join(Umap_folder, f"{case_id}_PADDED_img.nii.gz")
        confidence = os.path.join(Umap_folder, f"{case_id}_confidence.nii.gz")
        entropy = os.path.join(Umap_folder, f"{case_id}_entropy.nii.gz")
        mutual_info = os.path.join(Umap_folder, f"{case_id}_mutual_info.nii.gz")

        visualize_umap_gradcam_overlay(img_path, confidence, mask_path)
        visualize_umap_gradcam_overlay(img_path, entropy, mask_path)
        visualize_umap_gradcam_overlay(img_path, mutual_info, mask_path)

if __name__ == '__main__':
    main()


