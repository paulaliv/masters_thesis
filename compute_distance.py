import os
import sys
import glob
import numpy as np
from scipy.spatial import distance
import pandas as pd

def compute_train_dist():
    all_train_features = []
    feature_dirs = [
        "/gpfs/home6/palfken/ood_features/output",
        "/gpfs/home6/palfken/ood_features/output1"
    ]
    for folder in feature_dirs:
        for npz_file in glob.glob(os.path.join(folder,"*.npz")):
            data = np.load(npz_file)
            feats = data['features']  # shape: (num_patches, 320)
            all_train_features.append(feats)

    all_train_features = np.vstack(all_train_features)

    mean = np.mean(all_train_features, axis=0)  # shape: (320,)
    cov = np.cov(all_train_features, rowvar=False)  # shape: (320, 320)
    eps = 1e-6
    cov += eps * np.eye(cov.shape[0])
    cov_inv = np.linalg.inv(cov)
    return mean, cov, cov_inv

def save_train_distribution(mean, cov, cov_inv, filepath):
    np.savez(filepath, mean=mean, cov=cov, cov_inv=cov_inv)

def mahalanobis_distance(x, mean, cov_inv):
    delta = x - mean
    return np.sqrt(np.dot(np.dot(delta, cov_inv), delta.T))

def compute_test_dist(mean, cov_inv, test_feature_dir,csv_file):
    df = pd.read_csv(csv_file)

    all_distances = []
    subtypes_sorted = []

    for case_id in df['case_id']:
        feature_file = os.path.join(test_feature_dir, f"{case_id}_features.npz")

        # Load features for this case
        data = np.load(feature_file)
        test_features = data['features']  # shape: (num_patches, 320)

        # Compute distance per patch and average over patches for a case-level score
        distances = [mahalanobis_distance(x, mean, cov_inv) for x in test_features]

        #avg_distance = np.mean(distances)

        all_distances.append(distances)
        subtypes_sorted.append(df.loc[df['case_id'] == case_id, 'subtype'].values[0])

    all_distances = np.array(all_distances)
    return all_distances, subtypes_sorted


def main():
    mean, cov, cov_inv = compute_train_dist()

    print(f'MEAN: {mean}')
    print(f'COV: {cov}')
    print(f'COV_INV: {cov_inv}')

    save_loc = "/gpfs/home6/palfken/ood_features/"

    save_train_distribution(mean, cov, cov_inv, os.path.join(save_loc,"train_dist.npz"))

    # csv_file = "/path/to/ood_cases.csv"
    # test_feature_dir = "/path/to/test_features"
    #
    # distances, subtypes = compute_test_dist(mean, cov_inv, csv_file, test_feature_dir)
    #
    # # Now distances and subtypes arrays are aligned by case order
    # for dist, subtype in zip(distances, subtypes):
    #     print(f"Subtype: {subtype}, Mahalanobis distance: {dist}")
    #
if __name__ == "__main__":
    main()