#
import os
from collections import defaultdict
import torch
import numpy as np
import SimpleITK as sitk
import pandas as pd
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

#
#
# def get_mask_bounding_box_size(mask_array):
#     coords = np.where(mask_array == 1)
#     if len(coords[0]) == 0:
#         return None  # empty mask
#     z_min, y_min, x_min = np.min(coords[0]), np.min(coords[1]), np.min(coords[2])
#     z_max, y_max, x_max = np.max(coords[0]), np.max(coords[1]), np.max(coords[2])
#     size = (z_max - z_min + 1, y_max - y_min + 1, x_max - x_min + 1)
#     return size
#
# def bbox_size(mask, pad=0):
#     """Return padded (z,y,x) size or None if empty."""
#     coords = np.where(mask == 1)
#     if coords[0].size == 0:
#         return None
#     zmin, ymin, xmin = np.min(coords, axis=1)
#     zmax, ymax, xmax = np.max(coords, axis=1)
#     zmin = max(zmin - pad, 0); ymin = max(ymin - pad, 0); xmin = max(xmin - pad, 0)
#     zmax = min(zmax + pad, mask.shape[0] - 1)
#     ymax = min(ymax + pad, mask.shape[1] - 1)
#     xmax = min(xmax + pad, mask.shape[2] - 1)
#     return (zmax - zmin + 1, ymax - ymin + 1, xmax - xmin + 1)
#
# def compute_stats_from_sizes(sizes):
#     sizes = np.array(sizes)
#     return {
#         "mean": np.mean(sizes, axis=0),
#         "std": np.std(sizes, axis=0),
#         "min": np.min(sizes, axis=0),
#         "max": np.max(sizes, axis=0),
#         "p95": np.percentile(sizes, 95, axis=0),
#         "p98": np.percentile(sizes, 98, axis=0)
#     }
#
# def stats(arr3d):
#     arr = np.array(arr3d)
#     return dict(
#         mean=arr.mean(axis=0),
#         std=arr.std(axis=0),
#         min=arr.min(axis=0),
#         max=arr.max(axis=0),
#         p95=np.percentile(arr, 95, axis=0),
#         p98=np.percentile(arr, 98, axis=0)
#     )
# def diag(v):  # length of 3‑D size vector
#     return np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
#
# def print_stats(label, stats):
#     print(f"\nStats for {label}:")
#     print(f"  Mean bounding box size (z,y,x): {stats['mean']}")
#     print(f"  Std bounding box size (z,y,x): {stats['std']}")
#     print(f"  Min bounding box size (z,y,x): {stats['min']}")
#     print(f"  Max bounding box size (z,y,x): {stats['max']}")
#     print(f"  95 percentile (z,y,x): {stats['p95']}")
#     print(f"  98 percentile (z,y,x): {stats['p98']}")
#
# def compute_mask_stats(mask_dir, subtype_df, mask_ext=".nii.gz"):
#     all_sizes = []
#     subtype_sizes = defaultdict(list)
#
#     filenames = [f for f in os.listdir(mask_dir) if f.endswith(mask_ext)]
#     print(f"Found {len(filenames)} masks.")
#
#     for f in filenames:
#         stem = f.replace('.nii.gz', '')
#         mask_path = os.path.join(mask_dir, f)
#         mask_itk = sitk.ReadImage(mask_path)
#         mask_np = sitk.GetArrayFromImage(mask_itk)  # shape: [z,y,x]
#
#         # Get subtype
#         patient = subtype_df[subtype_df['nnunet_id'] == stem]
#         if not patient.empty:
#             subtype_label = patient['Final_Classification'].values[0]
#         else:
#             subtype_label = "Unknown"
#
#         # Binary mask
#         mask_bin = (mask_np > 0).astype(np.uint8)
#         bbox_size = get_mask_bounding_box_size(mask_bin)
#         if bbox_size is not None:
#             all_sizes.append(bbox_size)
#             subtype_sizes[subtype_label].append(bbox_size)
#         else:
#             print(f"Warning: Empty mask for file {f}")
#
#     # Global stats
#     global_stats = compute_stats_from_sizes(all_sizes)
#     print_stats("GLOBAL", global_stats)
#
#     # Per-subtype stats
#     for subtype_label, sizes in subtype_sizes.items():
#         stats = compute_stats_from_sizes(sizes)
#         print_stats(subtype_label, stats)
#
#     def diag(v):  # length of 3‑D size vector
#         return np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
#
# # ---------- main ----------
# def compute_mask_stats_with_ranking(mask_dir, df, mask_ext=".nii.gz", pad=0):
#
#     global_sizes, subtype_sizes = [], defaultdict(list)
#
#     for fn in sorted(f for f in os.listdir(mask_dir) if f.endswith(mask_ext)):
#         stem = fn.replace(mask_ext, "").replace(".nii", "")
#         subtype = df.loc[df['nnunet_id'] == stem, 'Final_Classification'].iloc[0] \
#             if any(df['nnunet_id'] == stem) else "Unknown"
#
#         arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_dir, fn)))
#         size = bbox_size((arr > 0).astype(np.uint8), pad=pad)
#         if size:
#             global_sizes.append(size);
#             subtype_sizes[subtype].append(size)
#
#     # -------- print global --------
#     gstats = stats(global_sizes)
#     print("\nGLOBAL stats");
#     [print(f"  {k}: {v}") for k, v in gstats.items()]
#
#     # -------- per‑subtype --------
#     subtype_table = []
#     for stype, lst in subtype_sizes.items():
#         st = stats(lst)
#         subtype_table.append(dict(
#             subtype=stype,
#             mean_diag=diag(st['mean']),
#             std_diag=diag(st['std']),
#             samples=len(lst),
#             **{f"{k}_{axis}": st[k][i] for k in ['mean', 'std', 'min', 'max', 'p95', 'p98']
#                for i, axis in enumerate(['z', 'y', 'x'])}
#         ))
#         # optional: print each subtype block
#         # print(f"\n{stype} ({len(lst)} cases)"); [print(f"  {k}: {v}") for k,v in st.items()]
#
#     tab = pd.DataFrame(subtype_table)
#
#     # -------- ranking --------
#     rank_mean = tab.sort_values('mean_diag', ascending=False)[['subtype', 'samples', 'mean_diag']]
#     rank_std = tab.sort_values('std_diag', ascending=False)[['subtype', 'samples', 'std_diag']]
#
#     print("\n=== Rank by MEAN diagonal size (largest first) ===")
#     print(rank_mean.to_string(index=False, formatters={'mean_diag': '{:.2f}'.format}))
#
#     print("\n=== Rank by STD diagonal size (highest variability first) ===")
#     print(rank_std.to_string(index=False, formatters={'std_diag': '{:.2f}'.format}))
#
#     return tab, gstats

if __name__ == "__main__":
    # mask_dir = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnUNet_raw/Dataset002_SoftTissue/labelsTr'
    # mask_dir_Ts = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnUNet_raw/Dataset002_SoftTissue/labelsTs'
    # tabular_data_dir = r'/home/bmep/plalfken/my-scratch/Downloads/WORC_data/WORC_all_with_nnunet_ids.csv'
    # tabular_data = pd.read_csv(tabular_data_dir)
    # print(tabular_data.columns)
    # subtype = tabular_data[['nnunet_id', 'Final_Classification']]
    # #compute_mask_stats(mask_dir,subtype)
    # #compute_mask_stats_with_ranking(mask_dir, subtype)
    # compute_mask_stats_with_ranking(mask_dir_Ts,subtype)
    image_shapes = []

    data_dir = r"/gpfs/home6/palfken/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/classification_Tr/"

    for fname in os.listdir(data_dir):
        if fname.endswith("_resized.pt"):
            case_id = fname.replace('_resized.pt', '')
            print(case_id)
            image = torch.load(os.path.join(data_dir, fname))

            image_shapes.append(image.shape)
        # Add other formats as needed

    # Convert to numpy array for easy stats
    shapes_array = np.array(image_shapes)
    # Print axis-wise mean, min, max sizes
    print("Number of samples:", len(shapes_array))
    print("Mean shape: ", np.mean(shapes_array, axis=0))
    print("Min shape:  ", np.min(shapes_array, axis=0))
    print("Max shape:  ", np.max(shapes_array, axis=0))
    print('95 Percentile', np.percentile(shapes_array, 93, axis =0))

'''

Stats for GLOBAL:
  Mean bounding box size (z,y,x): [ 17.40408163 114.31836735 110.9877551 ]
  Std bounding box size (z,y,x): [11.67851611 82.19621043 68.57895683]
  Min bounding box size (z,y,x): [1 9 9]
  Max bounding box size (z,y,x): [ 79 549 360]
  95 percentile (z,y,x): [ 40.8 251.8 240.6]
  98 percentile (z,y,x): [ 46.   305.24 300.6 ]

Stats for MyxofibroSarcomas :
  Mean bounding box size (z,y,x): [ 13.96491228 107.9122807  109.19298246]
  Std bounding box size (z,y,x): [ 8.33501368 63.01608837 63.99025107]
  Min bounding box size (z,y,x): [ 1 19  9]
  Max bounding box size (z,y,x): [ 40 281 281]
  95 percentile (z,y,x): [ 27.8 249.6 224.4]
  98 percentile (z,y,x): [ 38.04 259.92 240.16]

Stats for MyxoidlipoSarcoma:
  Mean bounding box size (z,y,x): [ 20.02857143 140.68571429 131.02857143]
  Std bounding box size (z,y,x): [ 9.11665889 86.15360748 74.94549856]
  Min bounding box size (z,y,x): [ 6 31 20]
  Max bounding box size (z,y,x): [ 46 463 360]
  95 percentile (z,y,x): [ 34.  265.8 281.7]
  98 percentile (z,y,x): [ 37.84 336.52 325.32]

Stats for DTF:
  Mean bounding box size (z,y,x): [12.77941176 67.19117647 78.27941176]
  Std bounding box size (z,y,x): [11.52853951 49.33297947 55.82979533]
  Min bounding box size (z,y,x): [ 1  9 12]
  Max bounding box size (z,y,x): [ 79 253 338]
  95 percentile (z,y,x): [ 28.6  170.5  163.25]
  98 percentile (z,y,x): [ 41.6  184.6  200.22]

Stats for LeiomyoSarcomas:
  Mean bounding box size (z,y,x): [12.87096774 90.74193548 92.83870968]
  Std bounding box size (z,y,x): [ 7.14252118 49.24197511 53.52063019]
  Min bounding box size (z,y,x): [ 2 32 22]
  Max bounding box size (z,y,x): [ 37 241 219]
  95 percentile (z,y,x): [ 24.5 170.  193. ]
  98 percentile (z,y,x): [ 29.8 199.  205.8]

Stats for WDLPS:
  Mean bounding box size (z,y,x): [ 27.75925926 176.87037037 151.5       ]
  Std bounding box size (z,y,x): [11.49063255 98.34351102 66.06443263]
  Min bounding box size (z,y,x): [ 6 56 40]
  Max bounding box size (z,y,x): [ 53 549 312]
  95 percentile (z,y,x): [ 46.   359.75 290.25]
  98 percentile (z,y,x): [ 48.82 456.84 304.7 ]
  
Stats for Lipoma:
  Mean bounding box size (z,y,x): [ 19.26315789 130.68421053 138.19298246]
  Std bounding box size (z,y,x): [ 7.57309716 78.96216236 80.87188018]
  Min bounding box size (z,y,x): [ 3 24 30]
  Max bounding box size (z,y,x): [ 40 340 404]
  95 percentile (z,y,x): [ 33.  298.2 280.8]
  98 percentile (z,y,x): [ 36.52 320.28 365.84]


GLOBAL stats
  mean: [ 17.40408163 114.31836735 110.9877551 ]
  std: [11.67851611 82.19621043 68.57895683]
  min: [1 9 9]
  max: [ 79 549 360]
  p95: [ 40.8 251.8 240.6]
  p98: [ 46.   305.24 300.6 ]

=== Rank by MEAN diagonal size (largest first) ===
           subtype  samples mean_diag
             WDLPS       54    234.53
 MyxoidlipoSarcoma       35    193.29
MyxofibroSarcomas        57    154.15
   LeiomyoSarcomas       31    130.46
               DTF       68    103.95

=== Rank by STD diagonal size (highest variability first) ===
           subtype  samples std_diag
             WDLPS       54   119.03
 MyxoidlipoSarcoma       35   114.55
MyxofibroSarcomas        57    90.20
               DTF       68    75.39
   LeiomyoSarcomas       31    73.08

Process finished with exit code 0

=== Rank by MEAN diagonal size (largest first) ===
           subtype  samples mean_diag
             WDLPS        3    342.93
            Lipoma       57    191.17
   LeiomyoSarcomas        2    169.05
 MyxoidlipoSarcoma        2    126.89
MyxofibroSarcomas         4    126.11
               DTF        4    113.94

=== Rank by STD diagonal size (highest variability first) ===
           subtype  samples std_diag
             WDLPS        3   203.36
            Lipoma       57   113.28
   LeiomyoSarcomas        2    80.02
 MyxoidlipoSarcoma        2    74.77
MyxofibroSarcomas         4    44.98
               DTF        4    34.49

'''