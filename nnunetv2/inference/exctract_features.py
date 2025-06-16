import inspect
import itertools
import multiprocessing
import os
from copy import deepcopy
from queue import Queue
from threading import Thread
from time import sleep
from typing import Tuple, Union, List, Optional

import numpy as np
import torch
import SimpleITK as sitk

from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import nnunetv2


from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def main():
    input_folder = "/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnUNet_raw/Dataset002_SoftTissue/imagesTr"
    output_folder = "/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/Logits_Tr"
    model_dir = "/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres"
    folds = (0,)
    # Create predictor
    predictor = nnUNetPredictor(
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )


    # Initialize from your trained model folder
    print('Initializing trained model ...')
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
    main()

