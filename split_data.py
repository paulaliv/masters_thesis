import os
import pandas as pd
import numpy
from glob import glob
import shutil
import json
from sklearn.model_selection import StratifiedKFold, train_test_split

fold_paths = {
    'fold_0': '/gpfs/home6/palfken/masters_thesis/fold_0',
    'fold_1': '/gpfs/home6/palfken/masters_thesis/fold_1',
    'fold_2': '/gpfs/home6/palfken/masters_thesis/fold_2',
    'fold_3': '/gpfs/home6/palfken/masters_thesis/fold_3',
    'fold_4': '/gpfs/home6/palfken/masters_thesis/fold_4',
}

input_folder= "/gpfs/home6/palfken/QA_dataTS_final"
output_folder = "/gpfs/home6/palfken/QA_dataTr_final"



#fold0= pd.read_csv(fold_paths['fold_0'])
#fold1= pd.read_csv(fold_paths['fold_1'])
fold2= pd.read_csv(fold_paths['fold_2'])
fold3= pd.read_csv(fold_paths['fold_3'])
fold4= pd.read_csv(fold_paths['fold_4'])

# Step 3: Merge remaining test cases with folds 2 and 3 to create training pool
cv_df = pd.concat([fold2, fold3, fold4], ignore_index=True)
case_ids = cv_df['case_id'].values
subtypes = cv_df['subtype'].values
# Prepare arrays for splitting


# Initialize StratifiedKFold with 5 splits, shuffle for randomness
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

splits = []

for train_idx, val_idx in skf.split(case_ids, subtypes):
    train_cases = case_ids[train_idx].tolist()
    print(f'number of train cases: {len(train_cases)}')
    val_cases = case_ids[val_idx].tolist()
    print(f'number of val cases {len(val_cases)}')
    splits.append({'train': train_cases, 'val': val_cases})

# Save to JSON file
output_path = '/gpfs/home6/palfken/QA_5fold_splits.json'
output_dir = os.path.dirname(output_path)

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Now write the JSON file
with open(output_path, 'w') as f:
    json.dump(splits, f, indent=4)

print("Stratified 5-fold splits saved to 'stratified_5fold_splits.json'")
def copy_cases(case_ids, input_folder,image_dst,):
    print(f'Moving {len(case_ids)} cases to {image_dst}')
    for case_id in case_ids:
        img_path = os.path.join(input_folder, f"{case_id}img.npy")
        confidence_path = os.path.join(input_folder, f"{case_id}_confidence.npy")
        mi_path = os.path.join(input_folder, f"{case_id}_mutual_info.npy")
        entropy_path = os.path.join(input_folder, f"{case_id}_entropy.npy")
        epkl_path = os.path.join(input_folder, f"{case_id}_epkl.npy")

        if os.path.exists(img_path):
            shutil.move(img_path, os.path.join(image_dst, os.path.basename(img_path)))
            shutil.move(confidence_path, os.path.join(image_dst, os.path.basename(confidence_path)))
            shutil.move(mi_path, os.path.join(image_dst, os.path.basename(mi_path)))
            shutil.move(entropy_path, os.path.join(image_dst, os.path.basename(entropy_path)))
            shutil.move(epkl_path, os.path.join(image_dst, os.path.basename(epkl_path)))
        else:
            print(f"Warning: Image not found for case {case_id}")

# # Copy files

# copy_cases(qa_train_ids, output_dirs["QA_imagesTr"], output_dirs["QA_labelsTr"])
# copy_cases(qa_test_ids, output_dirs["QA_imagesTs"], output_dirs["QA_labelsTs"])

# Loop over all files in source folder
initial_number =len(os.listdir(output_folder))
print(f'Number of files in source folder: {len(os.listdir(input_folder))}')

for filename in os.listdir(input_folder):
    src_file = os.path.join(input_folder, filename)
    dst_file = os.path.join(output_folder, filename)

    if os.path.isfile(src_file):  # Skip subfolders
        shutil.copy(src_file, dst_file)
new_number = len(os.listdir(output_folder))
print(f'new files in dst folder {new_number-initial_number} ')
