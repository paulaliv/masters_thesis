import os
import numpy as np
import torch
import json

from feature_visualization import cropped_mask
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

from torch.utils.hipify.hipify_python import meta_data


def save_metadata(metadata, save_path):
    # Option 1: Save as JSON (easier to read/edit manually)
    with open(save_path, 'w') as f:
        json.dump(metadata, f)

def save_metadata_npz(metadata, save_path):
    # Option 2: Save as npz (numpy compressed, fast to load)
    np.savez_compressed(save_path, **metadata)

def pad_or_crop(image, mask, target_shape):
    # image: tensor [C, D, H, W]
    # mask: tumor mask tensor same spatial size as image (for tumor location)
    # target_shape: (C, D, H, W)

    _, D, H, W = image.shape
    _, tD, tH, tW = target_shape

    print(f"Tumor voxels of mask = {np.sum(mask == 1)}")
    if mask.sum() == 0:
        print('empty mask')

    # Pad if smaller
    pad_d = max(tD - D, 0)
    pad_h = max(tH - H, 0)
    pad_w = max(tW - W, 0)

    # Pad symmetrically (can change if needed)
    pad = (pad_w // 2, pad_w - pad_w // 2,
           pad_h // 2, pad_h - pad_h // 2,
           pad_d // 2, pad_d - pad_d // 2)

    padding_info = pad
    crop_start = (0, 0, 0)
    # PyTorch expects pad for last dims in reverse order for F.pad:
    # (W_left, W_right, H_left, H_right, D_left, D_right)
    import torch.nn.functional as F
    if any(pad_i > 0 for pad_i in pad):
        image = F.pad(image, pad)
        mask = F.pad(mask, pad)

    # Crop if larger

    _, D, H, W = image.shape  # new shape after padding
    if any([D > tD, H > tH, W > tW]):
        crop_d = max(D - tD, 0)
        crop_h = max(H - tH, 0)
        crop_w = max(W - tW, 0)

        # Find tumor centroid along each dim
        tumor_indices = torch.nonzero(mask)
        centroid = tumor_indices.float().mean(dim=0) if tumor_indices.numel() > 0 else torch.tensor(
            [D // 2, H // 2, W // 2])
        cD, cH, cW = centroid.int()

        # Compute crop start positions ensuring crop window fits inside the image
        start_d = max(min(cD - tD // 2, D - tD), 0)
        start_h = max(min(cH - tH // 2, H - tH), 0)
        start_w = max(min(cW - tW // 2, W - tW), 0)

        crop_start = (start_d, start_h, start_w)
        # Crop image and mask
        image = image[:, start_d:start_d + tD, start_h:start_h + tH, start_w:start_w + tW]
        cropped_mask = mask[start_d:start_d + tD, start_h:start_h + tH, start_w:start_w + tW]

        print(f"Tumor voxels after final crop = {np.sum(cropped_mask == 1)}")
        if mask.sum() == 0 and cropped_mask.sum() == 0:
            print('empty mask')
        elif cropped_mask.sum() == 0 and mask.sum() > 0:
            print('Cropped out tumor')





    return image, crop_start, padding_info

def main():

    data_dir = r"/gpfs/home6/palfken/nnUNetFrame/nnunet_preprocessed/Dataset002_SoftTissue/nnUNetPlans_3d_fullres/"
    output_dir = "/gpfs/home6/palfken/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetPlans_3d_fullres/classification_Tr"
    target_shape = (1, 96, 576, 640)
    ds = nnUNetDatasetBlosc2(data_dir)
    for fname in os.listdir(data_dir):
        if fname.endswith(".b2nd") and not fname.endswith("_seg.b2nd"):
            case_id = fname.replace('.b2nd', '')
            print(case_id)
            data, seg, seg_prev, properties = ds.load_case(case_id)
            print("Data shape:", data.shape)

            image = data
            mask = seg

            resized_image, crop_start, padding_info = pad_or_crop(image, mask, target_shape )
            metadata = {
                "pad": list(padding_info),
                "crop_start": list(crop_start),
                "original_shape": list(data.shape)
            }


            image_file_name = f'{case_id}_resized.pt'
            meta_file_name = f'{case_id}_meta.json'
            save_metadata(metadata, os.path.join(output_dir, meta_file_name))
            torch.save(resized_image, os.path.join(output_dir, image_file_name))


    if __name__ == "__main__":
        main()