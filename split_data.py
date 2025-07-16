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

#nnunet_ids = pd.concat([fold0, fold1], ignore_index=True)
qa_train_ids = pd.concat([fold2, fold3], ignore_index=True)

# print(f'number of nnunet train cases ; {len(nnunet_ids)}')
print(f'number of qa train cases :  {len(qa_train_ids)}')
print(f'number of qa test cases ; {len(fold4)}')

# nnunet_case_ids = nnunet_ids['case_id'].unique()
qa_train_ids = qa_train_ids['case_id'].unique()
qa_test_ids = fold4['case_id'].unique()

# Step 2: Split fine-tuning val set from fold4 (qa_test)
tuning_val_df, remaining_test_df = train_test_split(
    fold4, test_size=0.8, stratify=fold4['subtype'], random_state=42
)

print(f"Fine-tuning validation cases: {len(tuning_val_df)}")
print(f"Remaining QA test cases: {len(remaining_test_df)}")

# Step 3: Merge remaining test cases with folds 2 and 3 to create training pool
cv_df = pd.concat([fold2, fold3, remaining_test_df], ignore_index=True)
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
with open('/gpfs/home6/palfken/QA_5fold_splits.json', 'w') as f:
    json.dump(splits, f, indent=4)

tuning_val_df.to_csv("gpfs/home6/palfken/fine_tuning_val_set.csv", index=False)


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
case_ids_move = remaining_test_df['case_id'].values


copy_cases(case_ids_move,input_folder, output_folder)
# copy_cases(qa_train_ids, output_dirs["QA_imagesTr"], output_dirs["QA_labelsTr"])
# copy_cases(qa_test_ids, output_dirs["QA_imagesTs"], output_dirs["QA_labelsTs"])
