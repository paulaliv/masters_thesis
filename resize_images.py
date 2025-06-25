import os
import numpy as np
import torch
import json

import torch.nn.functional as F

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

from torch.utils.hipify.hipify_python import meta_data


def save_metadata(metadata, save_path):
    # Option 1: Save as JSON (easier to read/edit manually)
    with open(save_path, 'w') as f:
        json.dump(metadata, f)

def save_metadata_npz(metadata, save_path):
    # Option 2: Save as npz (numpy compressed, fast to load)
    np.savez_compressed(save_path, **metadata)

def _smart_crop(volume, mask, target_shape):
        """
        Crop a C‑Z‑Y‑X volume (torch or np) and its mask to `target_shape`
        while guaranteeing that at least part of the tumour remains.

        Parameters
        ----------
        volume : np.ndarray | torch.Tensor
            Shape (C, Z, Y, X)
        mask   : np.ndarray  (binary)
            Shape (Z, Y, X)
        target_shape : tuple(int, int, int)
            Desired (dz, dy, dx)
        debug : bool
            If True prints crop indices and tumour stats.
        """
        # --- shapes ---
        C, Z, Y, X = volume.shape
        _,dz, dy, dx = target_shape

        # --- bounding box of tumour ---
        mask = mask.squeeze(0).cpu().numpy()
        zz, yy, xx = np.where(mask == 1)
        z_min, z_max = zz.min(), zz.max()
        y_min, y_max = yy.min(), yy.max()
        x_min, x_max = xx.min(), xx.max()

        def get_start(min_c, max_c, win, total):
            """Return start index so [start, start+win) keeps tumour inside."""
            size = max_c - min_c + 1
            if size >= win:  # tumour larger than window
                # keep centre of tumour inside
                centre = (min_c + max_c) // 2
                start = centre - win // 2
            else:
                # put tumour roughly in the middle, then adjust if near borders
                pad_left = (win - size) // 2
                pad_right = win - size - pad_left
                start = min_c - pad_left
            # clip to valid range
            return int(np.clip(start, 0, total - win))

        s_z = get_start(z_min, z_max, dz, Z)
        s_y = get_start(y_min, y_max, dy, Y)
        s_x = get_start(x_min, x_max, dx, X)

        crop_start = (s_z, s_y, s_x)

        # --- crop ---
        vol_crop = volume[:, s_z:s_z + dz, s_y:s_y + dy, s_x:s_x + dx]
        mask_crop = mask[s_z:s_z + dz, s_y:s_y + dy, s_x:s_x + dx]

        # --- safety fallback: if tumour lost, recrop around bbox front edge ---
        if mask_crop.sum() == 0:
            # Start so that bbox min corner is inside window
            s_z = int(np.clip(z_min, 0, Z - dz))
            s_y = int(np.clip(y_min, 0, Y - dy))
            s_x = int(np.clip(x_min, 0, X - dx))

            vol_crop = volume[:, s_z:s_z + dz, s_y:s_y + dy, s_x:s_x + dx]
            mask_crop = mask[s_z:s_z + dz, s_y:s_y + dy, s_x:s_x + dx]
            crop_start = (s_z, s_y, s_x)

        # if debug:
        #     print(f"bbox z [{z_min},{z_max}] → start_z={s_z}")
        #     print(f"bbox y [{y_min},{y_max}] → start_y={s_y}")
        #     print(f"bbox x [{x_min},{x_max}] → start_x={s_x}")
        #     print("Tumour voxels in crop =", mask_crop.sum())

        print(f"Tumor voxels of mask = {(mask_crop == 1).sum()}")

        if mask.sum() == 0 and mask_crop.sum() == 0:
            print('empty mask')
        elif mask_crop.sum() == 0 and mask.sum() > 0:
            print('Cropped out tumor')

        return vol_crop, mask_crop, crop_start

def pad_or_crop(image, mask, target_shape, context):
    # image: tensor [C, D, H, W]
    # mask: tumor mask tensor same spatial size as image (for tumor location)
    # target_shape: (C, D, H, W)

    _, D, H, W = image.shape
    _, tD, tH, tW = target_shape
    padding_info = (0, 0, 0, 0, 0, 0)
    crop_start = (0, 0, 0)

    print(f"Tumor voxels of mask = {(mask == 1).sum().item()}")

    if mask.sum() == 0:
        print('empty mask')

    coords = np.where(mask == 1)
    if coords[0].size == 0:
        print('Warning: empty mask, no tumor region found')
        # Pad if needed
        D, H, W = image.shape[1:]
        pad_d = max(tD - D, 0)
        pad_h = max(tH - H, 0)
        pad_w = max(tW - W, 0)

        padding_info = (
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2,
            pad_d // 2, pad_d - pad_d // 2
        )

        if any(p > 0 for p in padding_info):
            image = F.pad(image, padding_info)
            mask = F.pad(mask, padding_info)

        # Center crop if dimensions still too large
        _, D, H, W = image.shape
        s_z = max((D - tD) // 2, 0)
        s_y = max((H - tH) // 2, 0)
        s_x = max((W - tW) // 2, 0)

        cropped_image = image[:, s_z:s_z + tD, s_y:s_y + tH, s_x:s_x + tW]
        cropped_mask = mask[s_z:s_z + tD, s_y:s_y + tH, s_x:s_x + tW]
        crop_start = (s_z, s_y, s_x)
        bbox = {
            "z_min": None, "z_max": None,
            "y_min": None, "y_max": None,
            "x_min": None, "x_max": None,
            "z_min_m": None, "z_max_m": None,
            "y_min_m": None, "y_max_m": None,
            "x_min_m": None, "x_max_m": None
        }



    else:

        coords = torch.nonzero(mask, as_tuple=False)

        _,z_min, y_min, x_min = coords.min(dim=0).values
        _,z_max, y_max, x_max = coords.max(dim=0).values

        # Extend bbox by margin, clip min coordinates at 0 and max coordinates at largest valid index
        z_min_m = max(z_min.item() - context[0], 0)
        y_min_m = max(y_min.item() - context[1], 0)
        x_min_m = max(x_min.item() - context[2], 0)
        z_max_m = min(z_max.item() + context[0], mask.shape[0] - 1)
        y_max_m = min(y_max.item() + context[1], mask.shape[1] - 1)
        x_max_m = min(x_max.item() + context[2], mask.shape[2] - 1)

        bbox = {
            "z_min": z_min.item(),
            "z_max": z_max.item(),
            "y_min": y_min.item(),
            "y_max": y_max.item(),
            "x_min": x_min.item(),
            "x_max": x_max.item(),
            "z_min_m": z_min_m,
            "z_max_m": z_max_m,
            "y_min_m": y_min_m,
            "y_max_m": y_max_m,
            "x_min_m": x_min_m,
            "x_max_m": x_max_m}
        # Crop features

        cropped_image = image[:, z_min_m:z_max_m + 1, y_min_m:y_max_m + 1, x_min_m:x_max_m + 1]
        cropped_mask = mask[z_min_m:z_max_m + 1, y_min_m:y_max_m + 1, x_min_m:x_max_m + 1]

        # Pad if smaller
        D, H, W = cropped_image.shape[1:]
        pad_d = max(tD - D, 0)
        pad_h = max(tH - H, 0)
        pad_w = max(tW - W, 0)

        # Pad symmetrically (can change if needed)
        pad = (pad_w // 2, pad_w - pad_w // 2,
               pad_h // 2, pad_h - pad_h // 2,
               pad_d // 2, pad_d - pad_d // 2)

        padding_info = pad

        if any(pad_i > 0 for pad_i in pad):
            cropped_image = F.pad(image, pad)
            cropped_mask = F.pad(mask, pad)

    # Crop if larger

        _, D, H, W = cropped_image.shape  # new shape after padding
        if any([D > tD, H > tH, W > tW]):
            cropped_image, cropped_mask, crop_start = _smart_crop(cropped_image, cropped_mask, target_shape)



    return cropped_image, cropped_mask, crop_start, padding_info, bbox

def main():

    data_dir = r"/gpfs/home6/palfken/nnUNetFrame/nnunet_preprocessed/Dataset002_SoftTissue/nnUNetPlans_3d_fullres/"
    output_dir = r"/gpfs/home6/palfken/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/Classification_Tr/"


    target_shape = (1, 48, 272, 256)
    context = (3,15,15)

    ds = nnUNetDatasetBlosc2(data_dir)
    for fname in os.listdir(data_dir):
        if fname.endswith(".b2nd") and not fname.endswith("_seg.b2nd"):
            case_id = fname.replace('.b2nd', '')
            print(case_id)
            data, seg, seg_prev, properties = ds.load_case(case_id)
            print("Data shape:", data.shape)

            image = data
            mask = seg
            image = torch.tensor(np.array(image))
            mask = torch.tensor(np.array(mask))

            resized_image, resized_mask, crop_start, padding_info, bbox = pad_or_crop(image, mask, target_shape, context)

            metadata = {
                "pad": list(padding_info),  # tuple -> list
                "crop_start": list(crop_start),  # tensor -> list
                "original_shape": list(data.shape),# tuple -> list
                "bbox": bbox,
                "context":context
            }


            image_file_name = f'{case_id}_resized.pt'
            mask_file_name = f'{case_id}_mask_resized'
            meta_file_name = f'{case_id}_meta.json'

            save_metadata(metadata, os.path.join(output_dir, meta_file_name))
            torch.save(resized_image, os.path.join(output_dir, image_file_name))
            torch.save(resized_mask, os.path.join(output_dir, mask_file_name))

if __name__ == "__main__":
    main()