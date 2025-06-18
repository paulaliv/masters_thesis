import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

test1_dir = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/FeaturesTr/DES_0001_features_roi.npz'
test2_dir = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_results/Dataset002_SoftTissue/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/FeaturesTr/DES_0002_features_roi.npz'


def load_npz_array(path):
    with np.load(path) as data:
        return next(iter(data.values()))

def compute_stats(feat):
    print(f"Shape: {feat.shape}, dtype: {feat.dtype}")
    print(f"min: {feat.min()}, max: {feat.max()}, mean: {feat.mean()}, std: {feat.std()}")
    zero_ratio = np.sum(feat == 0) / feat.size
    print(f"Zero ratio: {zero_ratio:.4f}")


def flatten_features(feat):
    # Flatten all dimensions except batch/channels if needed
    return feat.reshape(feat.shape[0], -1)

feat1 = load_npz_array(test1_dir)
feat2 = load_npz_array(test2_dir)

print('DES_0001')
compute_stats(feat1)
print('DES_0002')
compute_stats(feat2)

# Flatten and compute cosine similarity channel-wise or feature-wise
f1_flat = flatten_features(feat1)
f2_flat = flatten_features(feat2)

# Compute cosine similarity per channel (rows)
# sklearn's cosine_similarity returns a matrix; diagonal entries are similarity between matching rows
cos_sim_matrix = cosine_similarity(f1_flat, f2_flat)

# We want similarity of matching channels (assumes same order)
diag_sim = np.diag(cos_sim_matrix)

print(f"Mean cosine similarity (per channel): {diag_sim.mean():.4f}")
print(f"Min cosine similarity (per channel): {diag_sim.min():.4f}")
print(f"Max cosine similarity (per channel): {diag_sim.max():.4f}")

