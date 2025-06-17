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



def compute_distances(features, mean, inv_cov):
    """
    Compute Mahalanobis distances for all rows in features.
    """
    return np.array([compute_mahalanobis_distance(x, mean, inv_cov) for x in features])


# def main():
#     # Prepare data: load features & labels
#     patient_ids = tabular_data['nnunet_id'].tolist()
#     labels = tabular_data['Final_Classification'].tolist()
#
#     all_features = []
#     for pid in patient_ids:
#         feats = load_features(pid)
#         flat_feats = flatten_feature_map(feats)
#         all_features.append(flat_feats)
#
#     # Concatenate all for global stats (all training data)
#     global_features = np.concatenate(all_features, axis=0)
#     mean_global, cov_global, inv_cov_global = compute_mean_cov_inv(global_features)
#
#     # Compute distances to global distribution
#     print("Distances to global distribution:")
#     for pid, feats in zip(patient_ids, all_features):
#         dists = compute_distances(feats, mean_global, inv_cov_global)
#         print(f"{pid}: mean distance {np.mean(dists):.3f}, std {np.std(dists):.3f}")
#
#     # Group by class and compute per-class stats
#     grouped = group_features_by_class(all_features, labels)
#     class_stats = {}
#     for cls, feats in grouped.items():
#         mean_cls, cov_cls, inv_cov_cls = compute_mean_cov_inv(feats)
#         class_stats[cls] = (mean_cls, inv_cov_cls)
#
#     # Example: For each patient, compute distance to each class distribution
#     print("\nDistances per class:")
#     for pid, feats in zip(patient_ids, all_features):
#         print(f"Patient {pid}:")
#         for cls, (mean_cls, inv_cov_cls) in class_stats.items():
#             dists = compute_distances(feats, mean_cls, inv_cov_cls)
#             print(f"  to class {cls}: mean dist = {np.mean(dists):.3f}")
def main():
    # Prepare data: load features & labels
    patient_ids = tabular_data['nnunet_id'].tolist()
    labels = tabular_data['Final_Classification'].tolist()

    all_features = []
    for file in os.listdir(feature_dir):
        patient_id = file.replace('features.npy', '')
        features, labels = load_patient_data(patient_id)
        flat_features = flatten(features)
        all_features.append(flat_features)

    # Concatenate all for global stats (all training data)
    global_features = np.concatenate(all_features, axis=0)
    mean_global, cov_global, inv_cov_global = compute_stats(global_features)

    # Compute distances to global distribution
    print("Distances to global distribution:")
    for pid, feats in zip(patient_ids, all_features):
        dists = compute_distances(feats, mean_global, inv_cov_global)
        print(f"{pid}: mean distance {np.mean(dists):.3f}, std {np.std(dists):.3f}")

    # Group by class and compute per-class stats
    grouped = group_features_by_class(all_features, labels)
    class_stats = {}
    for cls, feats in grouped.items():
        mean_cls, cov_cls, inv_cov_cls = compute_mean_cov_inv(feats)
        class_stats[cls] = (mean_cls, inv_cov_cls)

    # Example: For each patient, compute distance to each class distribution
    print("\nDistances per class:")
    for pid, feats in zip(patient_ids, all_features):
        print(f"Patient {pid}:")
        for cls, (mean_cls, inv_cov_cls) in class_stats.items():
            dists = compute_distances(feats, mean_cls, inv_cov_cls)
            print(f"  to class {cls}: mean dist = {np.mean(dists):.3f}")

if __name__ == "__main__":
    main()