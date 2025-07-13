import os
import pandas as pd
import numpy
from glob import glob
import shutil

fold_paths = {
    'fold_0': '/gpfs/home6/palfken/masters_thesis/fold_0',
    'fold_1': '/gpfs/home6/palfken/masters_thesis/fold_1',
    'fold_2': '/gpfs/home6/palfken/masters_thesis/fold_2',
    'fold_3': '/gpfs/home6/palfken/masters_thesis/fold_3',
    'fold_4': '/gpfs/home6/palfken/masters_thesis/fold_4',
}

input_folder_img = "/gpfs/home6/palfken/nnUNetFrame/nnUNet_raw/Dataset002_SoftTissue/imagesTr/"
input_folder_mask = "/gpfs/home6/palfken/nnUNetFrame/nnUNet_raw/Dataset002_SoftTissue/labelsTr/"

output_dirs = {
    "nnunet_imagesTr": "/gpfs/home6/palfken/nnUNetFrame/nnUNet_raw/Dataset002_SoftTissue/nnunet_imagesTr/",
    "nnunet_labelsTr": "/gpfs/home6/palfken/nnUNetFrame/nnUNet_raw/Dataset002_SoftTissue/nnunet_labelsTr/",
    "QA_imagesTr": "/gpfs/home6/palfken/QA_imagesTr/",
    "QA_labelsTr": "/gpfs/home6/palfken//QA_labelsTr/",
    "QA_imagesTs": "/gpfs/home6/palfken/QA_imagesTs/",
    "QA_labelsTs": "/gpfs/home6/palfken//QA_labelsTs/",

}

# Create output directories if they don't exist
for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

fold0= pd.read_csv(fold_paths['fold_0'])
fold1= pd.read_csv(fold_paths['fold_1'])
fold2= pd.read_csv(fold_paths['fold_2'])
fold3= pd.read_csv(fold_paths['fold_3'])
qa_test_ids= pd.read_csv(fold_paths['fold_4'])

nnunet_ids = pd.concat([fold0, fold1], ignore_index=True)
qa_train_ids = pd.concat([fold2, fold3], ignore_index=True)

print(f'number of nnunet train cases ; {len(nnunet_ids)}')
print(f'number of qa train cases :  {len(qa_train_ids)}')
print(f'number of qa test cases ; {len(qa_test_ids)}')

nnunet_case_ids = nnunet_ids['case_id'].unique()
qa_train_ids = qa_train_ids['case_id'].unique()
qa_test_ids = qa_test_ids['case_id'].unique()


def copy_cases(case_ids, image_dst, label_dst):
    for case_id in case_ids:

        img_path = os.path.join(input_folder_img, f"{case_id}_0000.nii.gz")
        label_path = os.path.join(input_folder_mask, f"{case_id}.nii.gz")

        if os.path.exists(img_path):
            shutil.copy(img_path, os.path.join(image_dst, os.path.basename(img_path)))
        else:
            print(f"Warning: Image not found for case {case_id}")

        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(label_dst, os.path.basename(label_path)))
        else:  #
            print(f"Warning: Image not found for case {case_id}")



# Copy files
copy_cases(nnunet_case_ids, output_dirs["nnunet_imagesTr"], output_dirs["nnunet_labelsTr"])
copy_cases(qa_train_ids, output_dirs["QA_imagesTr"], output_dirs["QA_labelsTr"])
copy_cases(qa_test_ids, output_dirs["QA_imagesTs"], output_dirs["QA_labelsTs"])
