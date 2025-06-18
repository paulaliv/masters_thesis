#
import os
import numpy as np
import SimpleITK as sitk


def get_mask_bounding_box_size(mask_array):
    coords = np.where(mask_array == 1)
    if len(coords[0]) == 0:
        # empty mask
        return None
    z_min, y_min, x_min = np.min(coords[0]), np.min(coords[1]), np.min(coords[2])
    z_max, y_max, x_max = np.max(coords[0]), np.max(coords[1]), np.max(coords[2])
    size = (z_max - z_min + 1, y_max - y_min + 1, x_max - x_min + 1)
    return size


def compute_mask_stats(mask_dir, mask_ext=".nii.gz"):
    sizes = []
    filenames = [f for f in os.listdir(mask_dir) if f.endswith(mask_ext)]
    print(f"Found {len(filenames)} masks.")

    for f in filenames:
        mask_path = os.path.join(mask_dir, f)
        mask_itk = sitk.ReadImage(mask_path)
        mask_np = sitk.GetArrayFromImage(mask_itk)  # shape: [z,y,x]

        # convert mask to binary if not already
        mask_bin = (mask_np > 0).astype(np.uint8)

        bbox_size = get_mask_bounding_box_size(mask_bin)
        if bbox_size is not None:
            sizes.append(bbox_size)
        else:
            print(f"Warning: Empty mask for file {f}")

    sizes = np.array(sizes)  # shape: (num_masks, 3)

    mean_size = np.mean(sizes, axis=0)
    std_size = np.std(sizes, axis=0)
    min_size = np.min(sizes, axis=0)
    max_size = np.max(sizes, axis=0)
    p95 = np.percentile(sizes, 95, axis=0)

    print(f"Mean bounding box size (z,y,x): {mean_size}")
    print(f"Std bounding box size (z,y,x): {std_size}")
    print(f"Min bounding box size (z,y,x): {min_size}")
    print(f"Max bounding box size (z,y,x): {max_size}")
    print(f"95 percentile (z,y,x): {p95}")




if __name__ == "__main__":
    mask_dir = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnUNet_raw/Dataset002_SoftTissue/labelsTr'

    #compute_mask_stats(mask_dir)
