import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

def get_bbox_size(mask):
    # Find non-zero voxels (tumor)
    coords = np.array(np.nonzero(mask))
    if coords.shape[1] == 0:
        return (0, 0, 0)  # empty mask

    # Get bounding box min and max
    min_coords = coords.min(axis=1)
    max_coords = coords.max(axis=1)

    # Get size in each dimension
    size = max_coords - min_coords + 1  # +1 because range is inclusive
    return tuple(size)

# Set your folder path here
mask_folder = "/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnUNet_raw/Dataset002_SoftTissue/labelsTr"

sizes = []

for filename in tqdm(os.listdir(mask_folder)):
    if not filename.endswith(".nii.gz"):
        continue

    mask_path = os.path.join(mask_folder, filename)
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()

    bbox_size = get_bbox_size(mask_data)
    sizes.append(bbox_size)

sizes = np.array(sizes)

# Print statistics
print("Number of masks:", len(sizes))
print("Min size:", sizes.min(axis=0))
print("Max size:", sizes.max(axis=0))
print("Mean size:", sizes.mean(axis=0).round(2))
print("90th percentile:", np.percentile(sizes, 90, axis=0).round(2))
