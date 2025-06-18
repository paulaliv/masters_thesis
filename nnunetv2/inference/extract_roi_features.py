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
import sys
sys.path.append('/gpfs/home6/palfken/masters_thesis/external/dynamic-network-architectures/dynamic_network_architectures')



def main(input_folder, output_folder, model_dir):
    folds = (0,1)
    # Create predictor
    predictor = nnUNetPredictor(
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),return_features = True
    ,)


    # Initialize from your trained model folder
    print('Initializing trained model ...')
    print(f'input folder: {input_folder}')
    print(f'output folder: {output_folder}')

    predictor.initialize_from_trained_model_folder(model_dir,
        use_folds=folds,   # or whatever fold(s) you want
        checkpoint_name='checkpoint_final.pth'
    )

    # This list will hold bottleneck features (one per image)
    #bottleneck_features = {}

    predictor.predict_from_files(
        list_of_lists_or_source_folder=input_folder,  # your folder with raw images
        output_folder_or_list_of_truncated_output_files=output_folder,  # where results get saved
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2
    )




if __name__ == '__main__':
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    model_dir = sys.argv[3]

    main(input_folder, output_folder, model_dir)
