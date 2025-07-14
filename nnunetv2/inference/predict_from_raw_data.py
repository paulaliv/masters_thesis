import inspect
import itertools
import multiprocessing
import os
from collections import defaultdict

import nibabel as nib
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
from sympy.solvers.diophantine.diophantine import reconstruct
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
import matplotlib.pyplot as plt
from skimage.transform import resize

import sys
import torch.nn.functional as F

sys.path.append('/gpfs/home6/palfken/masters_thesis/external/dynamic-network-architectures/dynamic_network_architectures')



class nnUNetPredictor(object):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = True,
                 return_features: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True):
        self.verbose = verbose

        self.return_features = return_features

        self.verbose_preprocessing = verbose_preprocessing
        self.allow_tqdm = allow_tqdm

        self.plans_manager, self.configuration_manager, self.list_of_parameters, self.network, self.dataset_json, \
        self.trainer_name, self.allowed_mirroring_axes, self.label_manager = None, None, None, None, None, None, None, None

        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        else:
            print(f'perform_everything_on_device=True is only supported for cuda devices! Setting this to False')
            perform_everything_on_device = False
        self.device = device
        self.perform_everything_on_device = perform_everything_on_device

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'), weights_only=False)
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None

            parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
                               f'Please place it there (in any .py file)!')
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters

        # initialize network with first set of parameters, also see https://github.com/MIC-DKFZ/nnUNet/issues/2520
        network.load_state_dict(parameters[0])

        self.network = network

        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    def manual_initialization(self, network: nn.Module, plans_manager: PlansManager,
                              configuration_manager: ConfigurationManager, parameters: Optional[List[dict]],
                              dataset_json: dict, trainer_name: str,
                              inference_allowed_mirroring_axes: Optional[Tuple[int, ...]]):
        """
        This is used by the nnUNetTrainer to initialize nnUNetPredictor for the final validation
        """
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        allow_compile = True
        allow_compile = allow_compile and ('nnUNet_compile' in os.environ.keys()) and (
                    os.environ['nnUNet_compile'].lower() in ('true', '1', 't'))
        allow_compile = allow_compile and not isinstance(self.network, OptimizedModule)
        if isinstance(self.network, DistributedDataParallel):
            allow_compile = allow_compile and isinstance(self.network.module, OptimizedModule)
        if allow_compile:
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    @staticmethod
    def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
        print('use_folds is None, attempting to auto detect available folds')
        fold_folders = subdirs(model_training_output_dir, prefix='fold_', join=False)
        fold_folders = [i for i in fold_folders if i != 'fold_all']
        fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
        use_folds = [int(i.split('_')[-1]) for i in fold_folders]
        print(f'found the following folds: {use_folds}')
        return use_folds

    def _manage_input_and_output_lists(self, list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                       output_folder_or_list_of_truncated_output_files: Union[None, str, List[str]],
                                       folder_with_segs_from_prev_stage: str = None,
                                       overwrite: bool = True,
                                       part_id: int = 0,
                                       num_parts: int = 1,
                                       save_probabilities: bool = False):
        if isinstance(list_of_lists_or_source_folder, str):
            list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(list_of_lists_or_source_folder,
                                                                                       self.dataset_json['file_ending'])
        print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
        list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]
        caseids = [os.path.basename(i[0])[:-(len(self.dataset_json['file_ending']) + 5)] for i in
                   list_of_lists_or_source_folder]
        print(
            f'I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
        print(f'There are {len(caseids)} cases that I would like to predict')

        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_filename_truncated = [join(output_folder_or_list_of_truncated_output_files, i) for i in caseids]
        else:
            output_filename_truncated = output_folder_or_list_of_truncated_output_files[part_id::num_parts]

        seg_from_prev_stage_files = [join(folder_with_segs_from_prev_stage, i + self.dataset_json['file_ending']) if
                                     folder_with_segs_from_prev_stage is not None else None for i in caseids]
        # remove already predicted files from the lists
        if not overwrite and output_filename_truncated is not None:
            tmp = [isfile(i + self.dataset_json['file_ending']) for i in output_filename_truncated]
            if save_probabilities:
                tmp2 = [isfile(i + '.npz') for i in output_filename_truncated]
                tmp = [i and j for i, j in zip(tmp, tmp2)]
            not_existing_indices = [i for i, j in enumerate(tmp) if not j]

            output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
            list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_existing_indices]
            seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in not_existing_indices]
            print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
                  f'That\'s {len(not_existing_indices)} cases.')
        return list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files

    def predict_from_files(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           num_processes_preprocessing: int = default_num_processes,
                           num_processes_segmentation_export: int = default_num_processes,
                           folder_with_segs_from_prev_stage: str = None,
                           num_parts: int = 1,
                           part_id: int = 0):
        """
        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        assert part_id <= num_parts, ("Part ID must be smaller than num_parts. Remember that we start counting with 0. "
                                      "So if there are 3 parts then valid part IDs are 0, 1, 2")
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            maybe_mkdir_p(output_folder)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(list_of_lists_or_source_folder,
                                                                                 seg_from_prev_stage_files,
                                                                                 output_filename_truncated,
                                                                                 num_processes_preprocessing)

        return self.predict_from_data_iterator(data_iterator, save_probabilities, num_processes_segmentation_export)

    def _internal_get_data_iterator_from_lists_of_filenames(self,
                                                            input_list_of_lists: List[List[str]],
                                                            seg_from_prev_stage_files: Union[List[str], None],
                                                            output_filenames_truncated: Union[List[str], None],
                                                            num_processes: int):
        return preprocessing_iterator_fromfiles(input_list_of_lists, seg_from_prev_stage_files,
                                                output_filenames_truncated, self.plans_manager, self.dataset_json,
                                                self.configuration_manager, num_processes, self.device.type == 'cuda',
                                                self.verbose_preprocessing)
        # preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose_preprocessing)
        # # hijack batchgenerators, yo
        # # we use the multiprocessing of the batchgenerators dataloader to handle all the background worker stuff. This
        # # way we don't have to reinvent the wheel here.
        # num_processes = max(1, min(num_processes, len(input_list_of_lists)))
        # ppa = PreprocessAdapter(input_list_of_lists, seg_from_prev_stage_files, preprocessor,
        #                         output_filenames_truncated, self.plans_manager, self.dataset_json,
        #                         self.configuration_manager, num_processes)
        # if num_processes == 0:
        #     mta = SingleThreadedAugmenter(ppa, None)
        # else:
        #     mta = MultiThreadedAugmenter(ppa, None, num_processes, 1, None, pin_memory=pin_memory)
        # return mta

    def get_data_iterator_from_raw_npy_data(self,
                                            image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                            segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,
                                                                                                        np.ndarray,
                                                                                                        List[
                                                                                                            np.ndarray]],
                                            properties_or_list_of_properties: Union[dict, List[dict]],
                                            truncated_ofname: Union[str, List[str], None],
                                            num_processes: int = 3):

        list_of_images = [image_or_list_of_images] if not isinstance(image_or_list_of_images, list) else \
            image_or_list_of_images

        if isinstance(segs_from_prev_stage_or_list_of_segs_from_prev_stage, np.ndarray):
            segs_from_prev_stage_or_list_of_segs_from_prev_stage = [
                segs_from_prev_stage_or_list_of_segs_from_prev_stage]

        if isinstance(truncated_ofname, str):
            truncated_ofname = [truncated_ofname]

        if isinstance(properties_or_list_of_properties, dict):
            properties_or_list_of_properties = [properties_or_list_of_properties]

        num_processes = min(num_processes, len(list_of_images))
        pp = preprocessing_iterator_fromnpy(
            list_of_images,
            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
            properties_or_list_of_properties,
            truncated_ofname,
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_processes,
            self.device.type == 'cuda',
            self.verbose_preprocessing
        )

        return pp

    def predict_from_list_of_npy_arrays(self,
                                        image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                        segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,
                                                                                                    np.ndarray,
                                                                                                    List[
                                                                                                        np.ndarray]],
                                        properties_or_list_of_properties: Union[dict, List[dict]],
                                        truncated_ofname: Union[str, List[str], None],
                                        num_processes: int = 3,
                                        save_probabilities: bool = False,
                                        num_processes_segmentation_export: int = default_num_processes):
        iterator = self.get_data_iterator_from_raw_npy_data(image_or_list_of_images,
                                                            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
                                                            properties_or_list_of_properties,
                                                            truncated_ofname,
                                                            num_processes)
        return self.predict_from_data_iterator(iterator, save_probabilities, num_processes_segmentation_export)

    from collections import defaultdict

    def reconstruct_full_feature_volume_sparse(self, patch_features, patch_locations, tumor_mask_shape):
        C = patch_features.shape[1]
        D, H, W = tumor_mask_shape

        assert patch_features.shape[0] == len(
            patch_locations), "Mismatch between number of patches and number of locations"
        assert patch_features.ndim == 5, "patch_features must be of shape (N, C, d, h, w)"
        assert isinstance(tumor_mask_shape, (tuple, list)) and len(
            tumor_mask_shape) == 3, "tumor_mask_shape must be (D, H, W)"

        #create sparse dictionary
        #each key is a voxel (z, y, x) and maps to [sum_vector, count]
        voxel_dict = defaultdict(lambda: [np.zeros(C, dtype=np.float32), 0])  # (C,), count

        for i, (z, y, x) in enumerate(patch_locations):
            patch = patch_features[i]
            if isinstance(patch, torch.Tensor):
                patch = patch.cpu().numpy()
            patch = patch.astype(np.float32)

            d, h, w = patch.shape[1:]

            assert z + d <= D and y + h <= H and x + w <= W, f"Patch {i} goes out of volume bounds"

            for dz in range(d):
                for dy in range(h):
                    for dx in range(w):
                        zz, yy, xx = z + dz, y + dy, x + dx
                        voxel_dict[(zz, yy, xx)][0] += patch[:, dz, dy, dx] #accumulate features
                        voxel_dict[(zz, yy, xx)][1] += 1 #count overlap

        # Convert sparse dict to dense array
        full_feature_volume = np.zeros((C, D, H, W), dtype=np.float32)
        for (zz, yy, xx), (vals, count) in voxel_dict.items():
            full_feature_volume[:, zz, yy, xx] = vals / count

        return full_feature_volume

    # def reconstruct_full_feature_volume(self, patch_features, patch_locations, tumor_mask_shape):
    #     """
    #     Reconstruct full feature volume from per patch features and their locations.
    #         Needed for extraction of ROI- feature map and potentially visualization
    #     """
    #     C = patch_features.shape[1]
    #     D, H, W = tumor_mask_shape
    #
    #     full_feature_volume = np.zeros((C, D, H, W), dtype=np.float32)
    #     count_map = np.zeros((D, H, W), dtype=np.float32)
    #
    #     for i, (z, y, x) in enumerate(patch_locations):
    #         patch = patch_features[i]  # (C, d, h, w)
    #         # Ensure it's a NumPy array
    #         if isinstance(patch, torch.Tensor):
    #             patch = patch.cpu().numpy()
    #
    #         # Ensure it's float32 (for addition)
    #         patch = patch.astype(np.float32)
    #
    #         d, h, w = patch.shape[1:]  # should match tile_size
    #
    #         full_feature_volume[:, z:z + d, y:y + h, x:x + w] += patch
    #         #counts overlapping patches per voxel
    #         count_map[z:z + d, y:y + h, x:x + w] += 1
    #
    #     # Avoid division by zero
    #     count_map[count_map == 0] = 1
    #     #averaging at each voxel -> corrects for overlapping regions
    #     full_feature_volume = full_feature_volume / count_map
    #
    #     assert full_feature_volume.shape[1:] == tumor_mask_shape, \
    #         f"Shape mismatch: features {full_feature_volume.shape[1:]}, mask {tumor_mask_shape}"
    #
    #     return full_feature_volume

    def _smart_crop(self, volume, mask, target_shape, debug=False):
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
        Z, Y, X = volume.shape
        dz, dy, dx = target_shape

        # --- bounding box of tumour ---
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

        # --- crop ---
        vol_crop = volume[s_z:s_z + dz, s_y:s_y + dy, s_x:s_x + dx]
        mask_crop = mask[s_z:s_z + dz, s_y:s_y + dy, s_x:s_x + dx]

        # --- safety fallback: if tumour lost, recrop around bbox front edge ---
        if mask_crop.sum() == 0:
            # Start so that bbox min corner is inside window
            s_z = int(np.clip(z_min, 0, Z - dz))
            s_y = int(np.clip(y_min, 0, Y - dy))
            s_x = int(np.clip(x_min, 0, X - dx))

            vol_crop = volume[s_z:s_z + dz, s_y:s_y + dy, s_x:s_x + dx]
            mask_crop = mask[s_z:s_z + dz, s_y:s_y + dy, s_x:s_x + dx]

        if debug:
            print(f"bbox z [{z_min},{z_max}] → start_z={s_z}")
            print(f"bbox y [{y_min},{y_max}] → start_y={s_y}")
            print(f"bbox x [{x_min},{x_max}] → start_x={s_x}")
            print("Tumour voxels in crop =", mask_crop.sum())

        assert mask_crop.sum() > 0, "Tumour lost after smart crop!"

        return vol_crop, mask_crop



    def reconstruct_and_crop_features(self,
                                      full_feature_volume,
                                      tumor_mask,
                                      uniform_size,
                                      context
                                      ):
        """
           Crops and pads feature map and tumor mask to uniform size centered around tumor.

           Args:
               feature_map: torch.Tensor of shape (C, Z, Y, X)
               tumor_mask: numpy array of shape (Z, Y, X)
               uniform_size: desired output shape (Z, Y, X)
               margin: number of voxels to extend beyond tumor bbox

           Returns:
               cropped_features: torch.Tensor of shape (C, Z, Y, X)
               cropped_mask: np.ndarray of shape (Z, Y, X)
           """


        # Tumor bounding box
        print(f'original tumor mask shape is {tumor_mask.shape}')
        print(f'original feature map size is {full_feature_volume.shape}')
        print(f'uniform size is {uniform_size}')

        # right after you compute tumor_mask
        unique = np.unique(tumor_mask)
        print("Unique labels in tumor_mask:", unique)
        print("Tumor voxels =", (tumor_mask != 0).sum())

        coords = np.where(tumor_mask == 1)
        if coords[0].size == 0:
            print('Warning: empty mask, no tumor region found')
            return np.zeros((full_feature_volume.shape[0], *uniform_size), dtype=np.float32), None

        else:
            z_min, y_min, x_min = np.min(coords[0]), np.min(coords[1]), np.min(coords[2])
            z_max, y_max, x_max = np.max(coords[0]), np.max(coords[1]), np.max(coords[2])

            #Extend bbox by margin, clip min coordinates at 0 and max coordinates at largest valid index
            z_min_m = max(z_min - context[0], 0)
            y_min_m = max(y_min - context[1], 0)
            x_min_m = max(x_min - context[2], 0)
            z_max_m = min(z_max + context[0], tumor_mask.shape[0] - 1)
            y_max_m = min(y_max + context[1], tumor_mask.shape[1] - 1)
            x_max_m = min(x_max + context[2], tumor_mask.shape[2] - 1)

            # Create slices for cropping
            z_slice = slice(z_min_m, z_max_m + 1)
            y_slice = slice(y_min_m, y_max_m + 1)
            x_slice = slice(x_min_m, x_max_m + 1)

            # Crop features
            cropped_features = full_feature_volume[ z_slice, y_slice, x_slice]
            cropped_mask = tumor_mask[z_slice, y_slice, x_slice]

            print(f'shape of feature map after cropping {cropped_features.shape}')
            print(f'shape of mask after cropping {cropped_mask.shape}')

            # after cropping
            assert cropped_mask.sum() > 0, "Tumor lost during cropping!"
            unique = np.unique(cropped_mask)
            print("Unique labels in tumor_mask:", unique)
            print("Tumor voxels =", (cropped_mask != 0).sum())

            # ---------------- padding if too small ----------------
            final_z, final_y, final_x = cropped_mask.shape
            desired_z, desired_y, desired_x = uniform_size

            pad_z = max(desired_z - final_z, 0)
            pad_y = max(desired_y - final_y, 0)
            pad_x = max(desired_x - final_x, 0)

            # symmetric padding
            pad = [(pad_z // 2, pad_z - pad_z // 2),
                   (pad_y // 2, pad_y - pad_y // 2),
                   (pad_x // 2, pad_x - pad_x // 2)]

            if any(p > 0 for p in (pad_z, pad_y, pad_x)):
                cropped_mask = np.pad(cropped_mask, pad, mode='constant', constant_values=0)
                cropped_features = torch.nn.functional.pad(cropped_features,
                                                           (pad[2][0], pad[2][1], pad[1][0], pad[1][1], pad[0][0],
                                                            pad[0][1]),
                                                           mode='constant', value=0)

            print(f"Shape after padding: features = {cropped_features.shape}, mask = {cropped_mask.shape}")



            # If cropped features are larger than uniform size, do center crop with warning
            final_z, final_y, final_x = cropped_features.shape

            # ---------- final crop if padded volume is still larger ----------
            if any([final_z > desired_z, final_y > desired_y, final_x > desired_x]):
                print('Warning: ROI region is larger than uniform size, image will be cropped')
                cropped_features, cropped_mask = self._smart_crop(
                    cropped_features, cropped_mask,
                    (desired_z, desired_y, desired_x)
                )
                # # crop around tumour centre, not volume centre
                # tz, ty, tx = np.where(cropped_mask == 1)
                # cz_tum, cy_tum, cx_tum = int(tz.mean()), int(ty.mean()), int(tx.mean())
                #
                # # compute crop start indices so that tumour centre stays inside
                # start_z = min(max(cz_tum - desired_z // 2, 0), final_z - desired_z)
                # start_y = min(max(cy_tum - desired_y // 2, 0), final_y - desired_y)
                # start_x = min(max(cx_tum - desired_x // 2, 0), final_x - desired_x)
                #
                # cropped_features = cropped_features[:, start_z:start_z + desired_z,
                #          start_y:start_y + desired_y,
                #          start_x:start_x + desired_x]
                # cropped_mask = cropped_mask[start_z:start_z + desired_z,
                #               start_y:start_y + desired_y,
                #               start_x:start_x + desired_x]
            print(f"Final cropped feature shape: {cropped_features.shape}")
            print(f"Final cropped mask shape: {cropped_mask.shape}")
            print(f"Tumor voxels after final crop = {np.sum(cropped_mask == 1)}")

            assert cropped_mask.sum() > 0, "ERROR: Tumor lost after final cropping!"

            assert cropped_features.shape == (desired_z, desired_y, desired_x), f"Feature map shape mismatch: {cropped_features.shape}"
            assert cropped_mask.shape == (desired_z, desired_y, desired_x), f"Mask shape mismatch: {cropped_mask.shape}"
            # if np.sum(padded_mask) == 0:
            #     print("Warning: padded mask is empty — tumor might have been cropped out")
            if np.sum(cropped_mask) == 0:
                raise RuntimeError(
                    "Aborting: Padded tumor mask is empty — tumor may have been cropped out or not predicted.")

            # returns cropped and padded feature volume, and spatial info
            return cropped_features

    def kl_divergence(self,p, q):
        return torch.sum(p * (torch.log(p + 1e-8) - torch.log(q + 1e-8)), dim=0)  # [H, W, D]

    def predict_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = default_num_processes):
        """
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                if ofile is not None:
                    print(f'\nPredicting {os.path.basename(ofile)}:')
                else:
                    print(f'\nPredicting image of shape {data.shape}:')

                print(f'perform_everything_on_device: {self.perform_everything_on_device}')

                properties = preprocessed['data_properties']

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
                # npy files
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)


                if self.return_features:
                    features, prediction, patch_locations = self.predict_logits_from_preprocessed_data(data)
                    print(f'final prediction shape {prediction.shape}')
                    prediction = prediction.cpu()
                    features = features.cpu()

                    print(f"Feature shape is {features.shape}")

                    #niform_size = [208, 256, 48]
                    #uniform_size = (48, 272, 256)

                    uniform_size = [48, 272, 256]

                    context = [3, 15, 15]
                    tumor_mask = prediction.argmax(0).numpy()
                    roi_features, cropped_mask = self.reconstruct_and_crop_features(
                        full_feature_volume=features,
                        tumor_mask=tumor_mask, uniform_size=uniform_size, context=context)

                    #convert feature file to nifty
                    # feature_file = ofile + "features.npz"
                    # np.save(feature_file, features)
                    # print(f"Saved features to {feature_file}")

                    if cropped_mask is not None:

                        feature_file_roi = ofile + "_features_roi.npz"
                        np.savez_compressed(feature_file_roi, roi_features)
                        print(f"Saved features to {feature_file_roi}")

                        cropped_mask_file = ofile + "_cropped_mask.npz"
                        np.savez_compressed(cropped_mask_file, cropped_mask)
                        print(f"Saved features to {cropped_mask_file}")
                        # patch_locations = ofile + "patch_locations.npy"
                        # np.save(patch_locations, patch_locations)
                        # print(f'Saved patch locations to {patch_locations}')
                    else:
                        print('Model returned an empty mask for this one')




                else:
                    #currently ensemble predictions
                    prediction, ensemble_predictions = self.predict_logits_from_preprocessed_data(data)
                    prediction = prediction.cpu()
                    #ensemble_predictions = ensemble_predictions.cpu()
                # saving the raw logits as numpy arrays

                    logits = torch.stack(ensemble_predictions) # shape: [5, C, H, W, D]
                    T, C, H, W, D = logits.shape
                    assert logits.shape == (T, C, H, W, D), "Logits should be [T, C, H, W, D]"

                    probs = F.softmax(logits, dim=1)  # shape: [5, C, H, W, D]
                    mean_probs = probs.mean(dim=0)  # shape: [C, H, W, D]
                    assert mean_probs.shape == (C, H, W, D), "Mean probs should be [C, H, W, D]"

                    # Confidence: max probability across classes
                    confidence_map = torch.max(mean_probs, dim=0).values  # shape: [H, W, D]
                    assert confidence_map.shape == (H, W, D), "Confidence map should be [H, W, D]"
                    entropy_map = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=0)  # shape: [H, W, D]
                    assert entropy_map.shape == (H, W, D), "Entropy map should be [H, W, D]"

                    # Entropy of each prediction
                    entropy_per_model = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # shape: [5, H, W, D]
                    assert entropy_per_model.shape == (T, H, W, D), "Entropy per model should be [T, H, W, D]"
                    mean_entropy = entropy_per_model.mean(dim=0)  # shape: [H, W, D]
                    assert mean_entropy.shape == (H, W, D), "Mean entropy should be [H, W, D]"

                    # Mutual information = entropy(mean_probs) - mean(entropy)
                    mutual_info = entropy_map - mean_entropy  # shape: [H, W, D]
                    assert mutual_info.shape == (H, W, D), "Mutual information should be [H, W, D]"

                    epkl_map = torch.zeros_like(entropy_map)
                    T = probs.shape[0]

                    for i in range(T):
                        for j in range(T):
                            if i != j:
                                epkl_map += self.kl_divergence(probs[i], probs[j])

                    epkl_map /= T * (T - 1)
                    assert epkl_map.shape == (H, W, D), "EPKL should be [H, W, D]"

                    # raw_logits_file = ofile + "_raw_logits.npy"
                    # np.save(raw_logits_file, raw_logits)
                    # print(f"Saved raw logits to {raw_logits_file}")


                    # uniform_size = [48, 272, 256]
                    #
                    # context = [3, 15, 15]
                    # tumor_mask = prediction.argmax(0).numpy()
                    # cropped_confidence= self.reconstruct_and_crop_features(
                    #     full_feature_volume=confidence_map,
                    #     tumor_mask=tumor_mask, uniform_size=uniform_size, context=context)
                    # print(f'Cropped Confidence shape {cropped_confidence.shape}')
                    #
                    # cropped_entropy = self.reconstruct_and_crop_features(
                    #     full_feature_volume=entropy_map,
                    #     tumor_mask=tumor_mask, uniform_size=uniform_size, context=context)
                    #
                    # cropped_MI = self.reconstruct_and_crop_features(
                    #     full_feature_volume=mutual_info,
                    #     tumor_mask=tumor_mask, uniform_size=uniform_size, context=context)
                    #
                    # cropped_kl= self.reconstruct_and_crop_features(
                    #     full_feature_volume=epkl_map,
                    #     tumor_mask=tumor_mask, uniform_size=uniform_size, context=context)


                    confidence_map = confidence_map[None] if confidence_map.ndim == 3 else confidence_map
                    entropy_map = entropy_map[None] if entropy_map.ndim == 3 else entropy_map
                    mutual_info = mutual_info[None] if mutual_info.ndim == 3 else mutual_info
                    epkl_map = epkl_map[None] if epkl_map.ndim == 3 else epkl_map



                    # # Resample logits to original image shape
                    current_spacing = self.configuration_manager.spacing if \
                        len(self.configuration_manager.spacing) == \
                        len(properties['shape_after_cropping_and_before_resampling']) else \
                        [properties['spacing'][0], *self.configuration_manager.spacing]


                    confidence_reshaped = self.configuration_manager.resampling_fn_probabilities(
                        confidence_map,
                        properties['shape_after_cropping_and_before_resampling'],
                        current_spacing,
                        properties['spacing']
                    )
                    entropy_reshaped = self.configuration_manager.resampling_fn_probabilities(
                        entropy_map,
                        properties['shape_after_cropping_and_before_resampling'],
                        current_spacing,
                        properties['spacing']
                    )
                    mutual_info_reshaped = self.configuration_manager.resampling_fn_probabilities(
                        mutual_info,
                        properties['shape_after_cropping_and_before_resampling'],
                        current_spacing,
                        properties['spacing']
                    )
                    epkl_reshaped = self.configuration_manager.resampling_fn_probabilities(
                        epkl_map,
                        properties['shape_after_cropping_and_before_resampling'],
                        current_spacing,
                        properties['spacing']
                    )


                    # # Save resampled logits AFTER resampling
                    # resampled_confidence_file = ofile + "_resampled_confidence.npy"
                    # np.save(resampled_confidence_file, confidence_reshaped)
                    # print(f"Saved resampled confidence to {resampled_confidence_file}")
                    # # Save resampled logits AFTER resampling
                    # resampled_entropy_file = ofile + "_resampled_entropy.npy"
                    # np.save(resampled_entropy_file, entropy_reshaped)
                    # print(f"Saved resampled logits to {resampled_entropy_file}")
                    #
                    # resampled_mi_file = ofile + "_resampled_mi.npy"
                    # np.save(resampled_mi_file, mutual_info_reshaped)
                    # print(f"Saved resampled logits to {resampled_mi_file}")
                    #
                    #
                    #


                    np.savez_compressed(ofile + "_uncertainty_maps.npz",
                                        confidence=confidence_reshaped.cpu().numpy(),
                                        entropy=entropy_reshaped.cpu().numpy(),
                                        mutual_info=mutual_info_reshaped.cpu().numpy(),
                                        epkl=epkl_reshaped.cpu().numpy())
                if ofile is not None:
                    # this needs to go into background processes
                    # export_prediction_from_logits(prediction, properties, self.configuration_manager, self.plans_manager,
                    #                               self.dataset_json, ofile, save_probabilities)
                    print('sending off prediction to background worker for resampling and export')
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((prediction, properties, self.configuration_manager, self.plans_manager,
                              self.dataset_json, ofile, save_probabilities),)
                        )
                    )
                else:
                    # convert_predicted_logits_to_segmentation_with_correct_shape(
                    #             prediction, self.plans_manager,
                    #              self.configuration_manager, self.label_manager,
                    #              properties,
                    #              save_probabilities)

                    print('sending off prediction to background worker for resampling')
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape, (
                                (prediction, self.plans_manager,
                                 self.configuration_manager, self.label_manager,
                                 properties,
                                 save_probabilities),)
                        )
                    )
                if ofile is not None:
                    print(f'done with {os.path.basename(ofile)}')
                else:
                    print(f'\nDone with image of shape {data.shape}:')
            ret = [i.get()[0] for i in r]



        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()


        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret



    def predict_single_npy_array(self, input_image: np.ndarray, image_properties: dict,
                                 segmentation_previous_stage: np.ndarray = None,
                                 output_file_truncated: str = None,
                                 save_or_return_probabilities: bool = False):
        """
        WARNING: SLOW. ONLY USE THIS IF YOU CANNOT GIVE NNUNET MULTIPLE IMAGES AT ONCE FOR SOME REASON.


        input_image: Make sure to load the image in the way nnU-Net expects! nnU-Net is trained on a certain axis
                     ordering which cannot be disturbed in inference,
                     otherwise you will get bad results. The easiest way to achieve that is to use the same I/O class
                     for loading images as was used during nnU-Net preprocessing! You can find that class in your
                     plans.json file under the key "image_reader_writer". If you decide to freestyle, know that the
                     default axis ordering for medical images is the one from SimpleITK. If you load with nibabel,
                     you need to transpose your axes AND your spacing from [x,y,z] to [z,y,x]!
        image_properties must only have a 'spacing' key!
        """
        ppa = PreprocessAdapterFromNpy([input_image], [segmentation_previous_stage], [image_properties],
                                       [output_file_truncated],
                                       self.plans_manager, self.dataset_json, self.configuration_manager,
                                       num_threads_in_multithreaded=1, verbose=self.verbose)
        if self.verbose:
            print('preprocessing')
        dct = next(ppa)

        if self.verbose:
            print('predicting')
        predicted_logits = self.predict_logits_from_preprocessed_data(dct['data']).cpu()

        if self.verbose:
            print('resampling to original shape')
        if output_file_truncated is not None:
            export_prediction_from_logits(predicted_logits, dct['data_properties'], self.configuration_manager,
                                          self.plans_manager, self.dataset_json, output_file_truncated,
                                          save_or_return_probabilities)
        else:
            ret = convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits, self.plans_manager,
                                                                              self.configuration_manager,
                                                                              self.label_manager,
                                                                              dct['data_properties'],
                                                                              return_probabilities=
                                                                              save_or_return_probabilities)
            if save_or_return_probabilities:
                return ret[0], ret[1]
            else:
                return ret


    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
        TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!

        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
        SEE convert_predicted_logits_to_segmentation_with_correct_shape
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
        prediction = None
        features = None

        patch_positions = None
        all_predictions  = []
        all_full_feature_volumes = []
        for params in self.list_of_parameters:

            # messing with state dict names...
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)

            # why not leave prediction on device if perform_everything_on_device? Because this may cause the
            # second iteration to crash due to OOM. Grabbing that with try except cause way more bloated code than
            # this actually saves computation time
            if self.return_features:
                new_features, new_prediction, new_patch_positions  = self.predict_sliding_window_return_logits(data)
                new_prediction = new_prediction.to('cpu')
                new_features = new_features.to('cpu')
                torch.cuda.empty_cache()


                mask_shape = new_prediction.shape[1:]
                #print(f'mask shape: {mask_shape}')

                full_feature_volume = self.reconstruct_full_feature_volume_sparse(new_features, new_patch_positions, mask_shape)
                all_full_feature_volumes.append(full_feature_volume)


            else:
                new_prediction = self.predict_sliding_window_return_logits(data).to('cpu')
            all_predictions.append(new_prediction)



            if prediction is None:
                prediction = new_prediction
                if self.return_features:
                    if patch_positions is None:
                        patch_positions = new_patch_positions
                #prediction = self.predict_sliding_window_return_logits(data).to('cpu')
            else:
                prediction += new_prediction

        if len(self.list_of_parameters) > 1:
            prediction /= len(self.list_of_parameters)

            if self.return_features:
            # Average full reconstructed feature volumes
                all_full_feature_volumes = [torch.as_tensor(x) for x in all_full_feature_volumes]
                ensemble_features = sum(all_full_feature_volumes) / len(all_full_feature_volumes)

        else:
            if self.return_features:
                ensemble_features = all_full_feature_volumes[0]


        if self.verbose: print('Prediction done')
        torch.set_num_threads(n_threads)
        if self.return_features:
            return ensemble_features, prediction, patch_positions
        else:
            return prediction, all_predictions

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
        slicers = []
        if len(self.configuration_manager.patch_size) < len(image_size):
            assert len(self.configuration_manager.patch_size) == len(
                image_size) - 1, 'if tile_size has less entries than image_size, ' \
                                 'len(tile_size) ' \
                                 'must be one shorter than len(image_size) ' \
                                 '(only dimension ' \
                                 'discrepancy of 1 allowed).'
            steps = compute_steps_for_sliding_window(image_size[1:], self.configuration_manager.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is'
                                   f' {image_size}, tile_size {self.configuration_manager.patch_size}, '
                                   f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                                     zip((sx, sy), self.configuration_manager.patch_size)]]))
        else:
            steps = compute_steps_for_sliding_window(image_size, self.configuration_manager.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(
                f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {self.configuration_manager.patch_size}, '
                f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                  zip((sx, sy, sz), self.configuration_manager.patch_size)]]))
        return slicers

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        network_output = self.network(x)
        if isinstance(network_output, tuple):
            features, prediction = network_output
            #print(f"Original features shape: {features.shape}")
        else:
            features = None
            prediction = network_output

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                mirrored_input = torch.flip(x, axes)
                mirrored_output = self.network(mirrored_input)

                if isinstance(mirrored_output, tuple):
                    mirrored_features, mirrored_prediction = mirrored_output


                else:
                    mirrored_prediction = mirrored_output
                    mirrored_features = None

                prediction += torch.flip(mirrored_prediction, axes)
                #print(f'mirrored prediction shape {mirrored_prediction.shape}')
                #mirror features to match predictions
                # if self.return_features and features is not None and mirrored_features is not None:
                #
                #     features += torch.flip(mirrored_features, axes)
                #     print(f"Flipped mirrored features shape: {torch.flip(mirrored_features, axes).shape}")
               #prediction += torch.flip(self.network(torch.flip(x, axes)), axes)

            #Averaging over original and all augmentations for prediciton and features
            prediction /= (len(axes_combinations) + 1)

            # if self.return_features and features is not None:
            #     features /= (len(axes_combinations) + 1)
            #     print(f'averaged features shape {features.shape}')

        if self.return_features:
            return features, prediction
        else:
            return prediction

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')
        all_patch_features = []
        patch_positions = []

        def get_slice_starts(sl):
            return tuple(s.start if isinstance(s, slice) else s for s in sl)

        #print(f"Number of slicers: {len(slicers)}")
        def producer(d, slh, q):
            for s in slh:
                q.put((torch.clone(d[s][None], memory_format=torch.contiguous_format).to(self.device), s))
            q.put('end')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)
            queue = Queue(maxsize=2)
            t = Thread(target=producer, args=(data, slicers, queue))
            t.start()

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)


            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')

            print(f'running prediction: {len(slicers)} steps')

            with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
                while True:
                    item = queue.get()
                    if item == 'end':
                        queue.task_done()
                        break
                    workon, sl = item
                    if self.return_features:

                        features, prediction = self._internal_maybe_mirror_and_predict(workon)

                        #double check what is now being returned for features and prediction
                        if features is not None:
                            features = features.to(results_device)
                            prediction = prediction[0].to(results_device)
                            #can only append if patches are same size!
                            # print(f'Feature shape before concatenating: {features.shape}')
                            # print(f'Prediction shape: {prediction.shape}')

                            all_patch_features.append(features)
                            starts = get_slice_starts(sl)
                            #print(f'Whole slicer output {starts}')
                            z_start, y_start, x_start = starts[-3:]
                            patch_positions.append((z_start, y_start, x_start))
                            #print(f'slice location: {z_start}, {y_start}, {x_start}')
                        else:
                            print('Internal_maybe_mirror_and_predict not working')


                    else:
                        prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)
                    #prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)


                    if self.use_gaussian:
                        gaussian = gaussian.to(prediction.device)
                        prediction *= gaussian

                    # if prediction.ndim == 4:
                    #     prediction = prediction.unsqueeze(0)  #  shape is [1, 2, 40, 320, 320]



                    predicted_logits[sl] += prediction
                    n_predictions[sl[1:]] += gaussian
                    queue.task_done()
                    pbar.update()
            queue.join()

            # predicted_logits /= n_predictions
            torch.div(predicted_logits, n_predictions, out=predicted_logits)
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e

        if self.return_features:

            features_concat = torch.cat(all_patch_features, dim=0)

            return features_concat, predicted_logits, patch_positions
        else:
            return predicted_logits

    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        self.network.eval()
        print('Network initiated')

        empty_cache(self.device)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
        # and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
        # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

            if self.verbose:
                print(f'Input shape: {input_image.shape}')
                print("step_size:", self.tile_step_size)
                print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                       'constant', {'value': 0}, True,
                                                       None)

            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])
            if self.return_features:
                if self.perform_everything_on_device and self.device != 'cpu':
                    # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
                    try:
                        features, predicted_logits, patch_positions  = self._internal_predict_sliding_window_return_logits(data, slicers,self.perform_everything_on_device)
                        #predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, self.perform_everything_on_device)
                    except RuntimeError:
                        print(
                            'Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                        empty_cache(self.device)
                        features, predicted_logits, patch_positions = self._internal_predict_sliding_window_return_logits(data, slicers, False)
                        #predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
                else:
                    features, predicted_logits, patch_positions = self._internal_predict_sliding_window_return_logits(data, slicers, self.perform_everything_on_device)
                    #predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,self.perform_everything_on_device)
                empty_cache(self.device)
                # revert padding
                predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
            else:
                predicted_logits= self._internal_predict_sliding_window_return_logits(data,slicers,self.perform_everything_on_device)
                predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]

        if self.return_features:
            return features, predicted_logits, patch_positions
        else:
            return predicted_logits

    def predict_from_files_sequential(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           folder_with_segs_from_prev_stage: str = None):
        """
        Just like predict_from_files but doesn't use any multiprocessing. Slow, but sometimes necessary
        """
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
            if len(output_folder) == 0:  # just a file was given without a folder
                output_folder = os.path.curdir
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files_sequential).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, 0, 1,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        label_manager = self.plans_manager.get_label_manager(self.dataset_json)
        preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose)

        if output_filename_truncated is None:
            output_filename_truncated = [None] * len(list_of_lists_or_source_folder)
        if seg_from_prev_stage_files is None:
            seg_from_prev_stage_files = [None] * len(seg_from_prev_stage_files)

        ret = []
        for li, of, sps in zip(list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files):
            data, seg, data_properties = preprocessor.run_case(
                li,
                sps,
                self.plans_manager,
                self.configuration_manager,
                self.dataset_json
            )

            print(f'perform_everything_on_device: {self.perform_everything_on_device}')

            prediction = self.predict_logits_from_preprocessed_data(torch.from_numpy(data)).cpu()

            if of is not None:
                export_prediction_from_logits(prediction, data_properties, self.configuration_manager, self.plans_manager,
                  self.dataset_json, of, save_probabilities)
            else:
                ret.append(convert_predicted_logits_to_segmentation_with_correct_shape(prediction, self.plans_manager,
                     self.configuration_manager, self.label_manager,
                     data_properties,
                     save_probabilities))

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret


def predict_entry_point_modelfolder():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-m', type=str, required=True,
                        help='Folder in which the trained model is. Must have subfolders fold_X for the different '
                             'folds you trained')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', '--c', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = nnUNetPredictor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                allow_tqdm=not args.disable_progress_bar,
                                verbose_preprocessing=args.verbose)
    predictor.initialize_from_trained_model_folder(args.m, args.f, args.chk)
    predictor.predict_from_files(args.i, args.o, save_probabilities=args.save_probabilities,
                                 overwrite=not args.continue_prediction,
                                 num_processes_preprocessing=args.npp,
                                 num_processes_segmentation_export=args.nps,
                                 folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                 num_parts=1, part_id=0)


def predict_entry_point():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset with which you would like to predict. You can specify either dataset name or id')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Specify the plans in which the desired configuration is located. '
                             'Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='What nnU-Net trainer class was used for training? Default: nnUNetTrainer')
    parser.add_argument('-c', type=str, required=True,
                        help='nnU-Net configuration that should be used for prediction. Config must be located '
                             'in the plans specified with -p')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-num_parts', type=int, required=False, default=1,
                        help='Number of separate nnUNetv2_predict call that you will be making. Default: 1 (= this one '
                             'call predicts everything)')
    parser.add_argument('-part_id', type=int, required=False, default=0,
                        help='If multiple nnUNetv2_predict exist, which one is this? IDs start with 0 can end with '
                             'num_parts - 1. So when you submit 5 nnUNetv2_predict calls you need to set -num_parts '
                             '5 and use -part_id 0, 1, 2, 3 and 4. Simple, right? Note: You are yourself responsible '
                             'to make these run on separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    # slightly passive aggressive haha
    assert args.part_id < args.num_parts, 'Do you even read the documentation? See nnUNetv2_predict -h.'

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = nnUNetPredictor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                verbose_preprocessing=args.verbose,
                                allow_tqdm=not args.disable_progress_bar)
    predictor.initialize_from_trained_model_folder(
        model_folder,
        args.f,
        checkpoint_name=args.chk
    )
    predictor.predict_from_files(args.i, args.o, save_probabilities=args.save_probabilities,
                                 overwrite=not args.continue_prediction,
                                 num_processes_preprocessing=args.npp,
                                 num_processes_segmentation_export=args.nps,
                                 folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                 num_parts=args.num_parts,
                                 part_id=args.part_id)
    # r = predict_from_raw_data(args.i,
    #                           args.o,
    #                           model_folder,
    #                           args.f,
    #                           args.step_size,
    #                           use_gaussian=True,
    #                           use_mirroring=not args.disable_tta,
    #                           perform_everything_on_device=True,
    #                           verbose=args.verbose,
    #                           save_probabilities=args.save_probabilities,
    #                           overwrite=not args.continue_prediction,
    #                           checkpoint_name=args.chk,
    #                           num_processes_preprocessing=args.npp,
    #                           num_processes_segmentation_export=args.nps,
    #                           folder_with_segs_from_prev_stage=args.prev_stage_predictions,
    #                           num_parts=args.num_parts,
    #                           part_id=args.part_id,
    #                           device=device)



def main(input_folder, output_folder, model_dir):
    folds = (0,1,2,3,4)
    # Create predictor
    predictor = nnUNetPredictor(
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        return_features=False,
    )


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
        num_processes_preprocessing=4,
        num_processes_segmentation_export=4
    )




if __name__ == '__main__':
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    model_dir = sys.argv[3]

    main(input_folder, output_folder, model_dir)

# if __name__ == '__main__':
#
#     ########################## predict a bunch of files
#     from nnunetv2.paths import nnUNet_results, nnUNet_raw
#
#     predictor = nnUNetPredictor(
#         tile_step_size=0.5,
#         use_gaussian=True,
#         use_mirroring=True,
#         perform_everything_on_device=True,
#         device=torch.device('cuda', 0),
#         verbose=False,
#         verbose_preprocessing=False,
#         allow_tqdm=True
#     )
#     predictor.initialize_from_trained_model_folder(
#         join(nnUNet_results, 'Dataset004_Hippocampus/nnUNetTrainer_5epochs__nnUNetPlans__3d_fullres'),
#         use_folds=(0,),
#         checkpoint_name='checkpoint_final.pth',
#     )
#     # predictor.predict_from_files(join(nnUNet_raw, 'Dataset003_Liver/imagesTs'),
#     #                              join(nnUNet_raw, 'Dataset003_Liver/imagesTs_predlowres'),
#     #                              save_probabilities=False, overwrite=False,
#     #                              num_processes_preprocessing=2, num_processes_segmentation_export=2,
#     #                              folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
#     #
#     # # predict a numpy array
#     # from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
#     #
#     # img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTr/liver_63_0000.nii.gz')])
#     # ret = predictor.predict_single_npy_array(img, props, None, None, False)
#     #
#     # iterator = predictor.get_data_iterator_from_raw_npy_data([img], None, [props], None, 1)
#     # ret = predictor.predict_from_data_iterator(iterator, False, 1)
#
#     ret = predictor.predict_from_files_sequential(
#         [['/media/isensee/raw_data/nnUNet_raw/Dataset004_Hippocampus/imagesTs/hippocampus_002_0000.nii.gz'], ['/media/isensee/raw_data/nnUNet_raw/Dataset004_Hippocampus/imagesTs/hippocampus_005_0000.nii.gz']],
#         '/home/isensee/temp/tmp', False, True, None
#     )
#
#
