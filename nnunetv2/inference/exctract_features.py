import inspect
import itertools
import multiprocessing
import os


import numpy as np
import torch
import SimpleITK as sitk

import nnunetv2
import sys

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def main(input_folder, output_folder, model_dir):
    folds = (0,1,2,3,4)
    # Create predictor
    predictor = nnUNetPredictor(
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )


    # Initialize from your trained model folder
    print('Initializing trained model ...')
    print(f'input folder: {input_folder}')
    print(f'output folder: {output_folder}')
    for files in os.listdir(input_folder):
        print(files)

    predictor.initialize_from_trained_model_folder(model_dir,
        use_folds=folds,   # or whatever fold(s) you want
        checkpoint_name='checkpoint_final.pth'
    )

    # This list will hold bottleneck features (one per image)
    #bottleneck_features = {}

    predictor.predict_from_files(
        list_of_lists_or_source_folder=input_folder,  # your folder with raw images
        output_folder_or_list_of_truncated_output_files=output_folder,  # where results get saved
        save_probabilities=True,
        overwrite=True,
        num_processes_preprocessing=4,
        num_processes_segmentation_export=4
    )




if __name__ == '__main__':
    input_folder = os.environ.get('nnUNet_raw', sys.argv[1])
    output_folder = os.environ.get('nnUNet_results', sys.argv[2])
    model_dir = sys.argv[3]

    main(input_folder, output_folder, model_dir)
