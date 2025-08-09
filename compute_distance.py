import os
import sys
import glob
import numpy as np
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt





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

def compute_id_train_minmax_from_features(mean, cov_inv):
    all_train_features = []
    folder = "/gpfs/home6/palfken/ood_features/id_data/"

    for npz_file in glob.glob(os.path.join(folder, "*features.npz")):
        data = np.load(npz_file)
        feats = data['features']  # shape: (num_patches, 320)
        all_train_features.append(feats)

    all_train_features = np.vstack(all_train_features)

    diffs = all_train_features - mean
    dists = np.sqrt(np.sum(diffs @ cov_inv * diffs, axis=1))
    id_min = dists.min()
    id_max = dists.max()
    return id_min, id_max


def subject_level_score(patch_features,mean, cov_inv, id_train_min, id_train_max):
    """
    Compute subject-level OOD score from voxel-wise values.

    voxel_map: np.ndarray, 3D uncertainty/distance map
    mask: np.ndarray, binary predicted mask (same shape as voxel_map)
    id_train_min: float, min score from ID training data
    id_train_max: float, max score from ID training data

    Returns: float (normalized subject-level score)
    """

    # Compute distance per patch
    patch_distances = [mahalanobis_distance(feat, mean, cov_inv)
                       for feat in patch_features]


    # Check if any voxel has a positive value
    if np.sum(patch_distances) >0:
        mean_value =  np.mean(patch_distances)
    else:
        print('Map is empty')
        return 0.0

    print(f"mean_value: {mean_value}, id_train_min: {id_train_min}, id_train_max: {id_train_max}")

    # 3. Normalize: min to doubled max from ID data
    norm_score = (mean_value - id_train_min) / (2 * id_train_max - id_train_min)
    norm_score = np.clip(norm_score, 0, None)  # no negatives

    print(f"norm_score (before clip): {(mean_value - id_train_min) / (2 * id_train_max - id_train_min)}")
    print(f"norm_score (after clip): {norm_score}")

    return norm_score


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

from sklearn.metrics import roc_auc_score, roc_curve

def tpr_at_fpr(scores, labels, target_fpr=0.05):
    labels_binary = np.where(labels == 'OOD', 1, 0)
    fpr, tpr, thresholds = roc_curve(labels_binary, scores)
    # Find closest FPR index
    idx = np.argmin(np.abs(fpr - target_fpr))
    return tpr[idx], thresholds[idx]


def main():
    # mean, cov, cov_inv = compute_train_dist()
    #
    # print(f'MEAN: {mean}')
    # print(f'COV: {cov}')
    # print(f'COV_INV: {cov_inv}')
    #
    maps_dir =  "/gpfs/home6/palfken/ood_features/features"
    subtypes_csv = "/gpfs/home6/palfken/WORC_test.csv"
    subtypes_df = pd.read_csv(subtypes_csv)


    #
    # save_train_distribution(mean, cov, cov_inv, os.path.join(save_loc,"train_dist.npz"))
    train_data = np.load("/gpfs/home6/palfken/ood_features/train_dist.npz")
    mean, cov, cov_inv,id_min, id_max = train_data['mean'], train_data['cov'], train_data['cov_inv'], train_data['id_min'], train_data['id_max']
    scores = []
    labels = []

    for npz_file in glob.glob(os.path.join(maps_dir, "*features.npz")):
        case_id = os.path.basename(npz_file).replace('_features.npz', '')
        print(case_id)

        subtype_row = subtypes_df[subtypes_df['nnunet_id'] == case_id]
        if not subtype_row.empty:
            tumor_class = subtype_row.iloc[0]['Final_Classification']
            tumor_class = tumor_class.strip()
            print(f'Tumor type: {tumor_class}')
            if tumor_class == 'Lipoma':
                tumor_class = 'OOD'
            else:
                tumor_class = 'ID'
            print(f'Data label: {tumor_class}')
        else:
            tumor_class = 'Unknown'
            print(f'Case id {case_id}: no subtype in csv file!')


        data= np.load(npz_file)
        dist = data['features']


        score = subject_level_score(dist,mean, cov_inv, id_min, id_max)
        print(f'Score: {score}')

        scores.append(score)
        labels.append(tumor_class)

    labels_array = np.array(labels)

    auc = roc_auc_score(labels_array, scores)
    tpr95, threshold = tpr_at_fpr(scores, labels_array, 0.05)



    # Separate ID and OOD scores
    scores = np.array(scores)
    labels_array = np.array(labels)

    id_scores = scores[labels_array == 'ID']
    ood_scores = scores[labels_array == 'OOD']


    # Small jitter on x for visibility
    id_x = np.ones_like(id_scores) + np.random.uniform(-0.05, 0.05, size=len(id_scores))
    ood_x = 2 * np.ones_like(ood_scores) + np.random.uniform(-0.05, 0.05, size=len(ood_scores))

    plt.figure(figsize=(8, 6))
    plt.scatter(id_x, id_scores, label='ID', alpha=0.7, color='blue', marker='o')
    plt.scatter(ood_x, ood_scores, label='OOD', alpha=0.7, color='red', marker='x')

    plt.axhline(y=threshold, color='green', linestyle='--', label=f'Threshold={threshold:.3f}')

    plt.xticks([1, 2], ['ID', 'OOD'])
    plt.ylabel('OOD Score')
    plt.title('OOD Scores per Subject with Threshold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.savefig('/gpfs/home6/palfken/ood_features/ood_scatterplot.png')

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