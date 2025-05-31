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
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy, preprocessing_iterator_fromfiles, \
    preprocessing_iterator_fromnpy
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


# Create predictor
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=False,
    perform_everything_on_device=True,
    device=torch.device('cpu'),
)

# Initialize from your trained model folder
predictor.initialize_from_trained_model_folder(
    model_training_output_dir=r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres',
    use_folds=(0,),  # or whatever fold(s) you want
    checkpoint_name='checkpoint_final.pth'
)

test_image = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnUNet_raw/Dataset002_SoftTissue/imagesTs/DES_0020_0000.nii.gz'

predictor.network.eval()

with torch.no_grad():
    encoder_outputs = predictor.network.encoder(test_image, return_skips=True)
    bottleneck = encoder_outputs[-1]  # Last stage = bottleneck


torch.save(bottleneck.cpu(), "bottleneck_features.pt")
# or

np.save("/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/bottleneck_features.npy", bottleneck.cpu().numpy())
