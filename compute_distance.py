import numpy as np
from scipy.spatial import distance
from sympy.solvers.diophantine.diophantine import prime_as_sum_of_two_squares
import pandas as pd
# data containing patient id under nnunet_id and label under Final_Classification
tabular_data_dir = r'/home/bmep/plalfken/my-scratch/Downloads/WORC_data/WORC_all_with_nnunet_ids.csv'
tabular_data = pd.read_csv(tabular_data_dir)
# Feature maps stored i folder always with name patient_id_features.npy
feature_dir = r'/home/bmep/plalfken/my-scratch/Downloads'
subtype = tabular_data[['nnunet_id', 'Final_Classification']]

def load_patient_data(patient_id):
    """Load the features for a given patient."""
    file_path = os.path.join(feature_dir, f"{patient_id}_features.npy")


    patient = subtype[subtype['nnunet_id'] == patient_id]
    if not patient.empty:
        subtype_label = patient['Final_Classification'].values[0]

    return np.load(file_path), subtype_label  # Expecting e.g. shape (C, D, H, W) or similar


def flatten(feature_map):
    """
      Flatten spatial dimensions of feature map into 2D: (num_samples, feature_dim).
      Assumes feature_map shape (C, D, H, W) or (C, H, W).
      """
    if feature_map.ndim == 4:
        # e.g. (C, D, H, W)
        C, D, H, W = feature_map.shape
        return feature_map.reshape(C, D * H * W).T  # shape (num_samples, C)
    elif feature_map.ndim == 3:
        C, H, W = feature_map.shape
        return feature_map.reshape(C, H * W).T  # shape (num_samples, C)
    else:
        raise ValueError(f"Unsupported feature map shape: {feature_map.shape}")


def group_by_class(features, labels):
    """
       Group features by class label.
       features_list: list of (num_samples_i, feature_dim) arrays
       labels: corresponding list of class labels (one per feature array)

       Returns:
           dict: class_label -> concatenated features of that class (N_i, D)
       """
    grouped = {}
    for features, label in zip(features, labels):
        if label not in grouped:
            grouped[label] = []
        grouped[label].append(features)
    # Concatenate all features per class
    for k in grouped:
        grouped[k] = np.concatenate(grouped[k], axis=0)
    return grouped

def compute_stats(flattened_features):
    mean_vec = np.mean(flattened_features, axis=0)  # shape (D,)
    cov_matrix = np.cov(flattened_features, rowvar=False)  # shape (D, D)

    # Step 2: Invert covariance matrix
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    return mean_vec, cov_matrix, inv_cov_matrix


def compute_mahalanobis_distance(x, mean, inv_cov):
    delta = x - mean
    return np.sqrt(np.dot(np.dot(delta, inv_cov), delta.T))

# Step 1: Compute mean and covariance of training features
mean_vec = np.mean(train_features, axis=0)  # shape (D,)
cov_matrix = np.cov(train_features, rowvar=False)  # shape (D, D)

# Step 2: Invert covariance matrix
inv_cov_matrix = np.linalg.inv(cov_matrix)

# Step 3: Compute Mahalanobis distance for each test feature vector
def mahalanobis_dist(x, mean, inv_cov):
    delta = x - mean
    return np.sqrt(np.dot(np.dot(delta, inv_cov), delta.T))

distances = np.array([mahalanobis_dist(x, mean_vec, inv_cov_matrix) for x in test_features])
