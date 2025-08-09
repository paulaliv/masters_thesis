import numpy as np


def subject_level_score(voxel_map, mask, id_train_min, id_train_max):
    """
    Compute subject-level OOD score from voxel-wise values.

    voxel_map: np.ndarray, 3D uncertainty/distance map
    mask: np.ndarray, binary predicted mask (same shape as voxel_map)
    id_train_min: float, min score from ID training data
    id_train_max: float, max score from ID training data

    Returns: float (normalized subject-level score)
    """

    # 1. Select only voxels within mask
    masked_values = voxel_map[mask > 0]

    if len(masked_values) == 0:
        return 0.0  # no lesion

    # 2. Average voxel values
    mean_value = masked_values.mean()

    # 3. Normalize: min to doubled max from ID data
    norm_score = (mean_value - id_train_min) / (2 * id_train_max - id_train_min)
    norm_score = np.clip(norm_score, 0, None)  # no negatives

    return norm_score

