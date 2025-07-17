import os
import pandas as pd
import numpy
from glob import glob
import shutil
import json
from sklearn.model_selection import StratifiedKFold, train_test_split


# Step 3: Merge remaining test cases with folds 2 and 3 to create training pool

df_dir = '/home/bmep/plalfken/my-scratch/nnUNet/Final_dice_dist.csv'
df = pd.read_csv(df_dir)

import shutil

# Step 1: Filter case_ids starting with '20EP'
selected_cases = df[df['case_id'].str.startswith('20EP')]['case_id'].tolist()
print(selected_cases)

# Step 2: Define paths
input_folder = "/gpfs/home6/palfken/20QA_dataTr_final/"
output_folder = "/gpfs/home6/palfken/QA_dataTr_final/"

# Step 3: Move each case folder/file
for case_id in selected_cases:
    pattern = os.path.join(input_folder, case_id + '*')
    files = glob(pattern)
    if not files:
        print(f"No files found for {case_id}")
        continue
    for file in files:

        dst_path = os.path.join(output_folder, os.path.basename(file))

        if os.path.exists(file):
            shutil.copy(file, dst_path)
            print(f"Copied: {case_id}")
        else:
            print(f"Not found: {case_id}")

# print("Stratified 5-fold splits saved to 'stratified_5fold_splits.json'")
# def copy_cases(case_ids, input_folder,image_dst,):
#     print(f'Moving {len(case_ids)} cases to {image_dst}')
#     for case_id in case_ids:
#         img_path = os.path.join(input_folder, f"{case_id}img.npy")
#         confidence_path = os.path.join(input_folder, f"{case_id}_confidence.npy")
#         mi_path = os.path.join(input_folder, f"{case_id}_mutual_info.npy")
#         entropy_path = os.path.join(input_folder, f"{case_id}_entropy.npy")
#         epkl_path = os.path.join(input_folder, f"{case_id}_epkl.npy")
#
#         if os.path.exists(img_path):
#             shutil.move(img_path, os.path.join(image_dst, os.path.basename(img_path)))
#             shutil.move(confidence_path, os.path.join(image_dst, os.path.basename(confidence_path)))
#             shutil.move(mi_path, os.path.join(image_dst, os.path.basename(mi_path)))
#             shutil.move(entropy_path, os.path.join(image_dst, os.path.basename(entropy_path)))
#             shutil.move(epkl_path, os.path.join(image_dst, os.path.basename(epkl_path)))
#         else:
#             print(f"Warning: Image not found for case {case_id}")
#
# # # Copy files
#
# # copy_cases(qa_train_ids, output_dirs["QA_imagesTr"], output_dirs["QA_labelsTr"])
# # copy_cases(qa_test_ids, output_dirs["QA_imagesTs"], output_dirs["QA_labelsTs"])
#
# # Loop over all files in source folder
# initial_number =len(os.listdir(output_folder))
# print(f'Number of files in source folder: {len(os.listdir(input_folder))}')
#
# for filename in os.listdir(input_folder):
#     src_file = os.path.join(input_folder, filename)
#     dst_file = os.path.join(output_folder, filename)
#
#     if os.path.isfile(src_file):  # Skip subfolders
#         shutil.move(src_file, dst_file)
# new_number = len(os.listdir(output_folder))
# print(f'new files in dst folder {new_number-initial_number} ')
