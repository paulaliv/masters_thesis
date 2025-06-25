import os
import numpy as np
import torch
import json

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

        return vol_crop, crop_start

def pad_or_crop(image, mask, target_shape):
    # image: tensor [C, D, H, W]
    # mask: tumor mask tensor same spatial size as image (for tumor location)
    # target_shape: (C, D, H, W)

    _, D, H, W = image.shape
    _, tD, tH, tW = target_shape

    print(f"Tumor voxels of mask = {(mask == 1).sum().item()}")

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
        image, crop_start = _smart_crop(image, mask, target_shape)








    return image, crop_start, padding_info

def main():

    data_dir = r"/gpfs/home6/palfken/nnUNetFrame/nnunet_preprocessed/Dataset002_SoftTissue/nnUNetPlans_3d_fullres/"
    output_dir = r"/gpfs/home6/palfken/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/classification_Tr/"

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
            image = torch.tensor(np.array(image))
            mask = torch.tensor(np.array(mask))

            resized_image, crop_start, padding_info = pad_or_crop(image, mask, target_shape)

            metadata = {
                "pad": list(padding_info),  # tuple -> list
                "crop_start": list(crop_start),  # tensor -> list
                "original_shape": list(data.shape)  # tuple -> list
            }

            image_file_name = f'{case_id}_resized.pt'
            meta_file_name = f'{case_id}_meta.json'

            save_metadata(metadata, os.path.join(output_dir, meta_file_name))
            torch.save(resized_image, os.path.join(output_dir, image_file_name))


if __name__ == "__main__":
    main()