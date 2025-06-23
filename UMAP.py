#!/usr/bin/env python
"""
visualize_subtype_clusters.py

USAGE
-----
python visualize_subtype_clusters.py \
    --feature_dir_tr /path/to/feature_dir_Tr \
    --feature_dir_ts /path/to/feature_dir_Ts \
    --csv /home/bmep/plalfken/my-scratch/Downloads/WORC_data/WORC_all_with_nnunet_ids.csv

Each feature file must be named  <nnunet_id>_features.npy  (case‑insensitive).
The CSV must contain columns  'nnunet_id'  and  'Final_Classification'.
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import umap # pip install umap-learn
from collections import defaultdict
from tqdm import tqdm
from sklearn.decomposition import PCA


def run_pca(feature_dir):
    all_feats = []  # shape (N, C, D, H, W)

    pattern = os.path.join(feature_dir, "*_features_roi.npz")
    for file in os.listdir(feature_dir):
        if file.endswith("_features_roi.npz"):
            nn_id = file.replace('_features_roi.npz', '')  # 'STT_0001_features.npy' -> 'STT_0001'
            feat1 = np.load(file)
            feat = feat1[feat1.files[0]]
            print(feat.shape)
            C = feat.shape[0]
            averaged_feat = feat.reshape(C, -1).mean(axis=1)
            print(f'feature shape {averaged_feat.shape}')
            all_feats.append(averaged_feat)

    x = np.vstack(all_feats)  # shape: (N, C)
    pca = PCA()
    pca.fit(x)
    print("Explained variance per PC:", pca.explained_variance_ratio_)

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA - Channel-Level Feature Importance")
    plt.grid()
    plt.savefig(image_dir, dpi=300)


def load_feature_vectors(feature_dir: str) -> dict[str, np.ndarray]:
    """
    Reads each *_features.npy in a directory, performs global‑average pooling
    (mean over all spatial dims) so the result is a 1‑D vector (C,).

    Returns
    -------
    dict { nnunet_id (str) : 1‑D np.ndarray }
    """
    vectors = {}
    pattern = os.path.join(feature_dir, "*_features_roi.npz")
    for path in tqdm(sorted(glob.glob(pattern)), desc=f"Loading from {feature_dir}"):
        fname = os.path.basename(path)
        nn_id = fname.replace('_features_roi.npz','') # 'STT_0001_features.npy' -> 'STT_0001'
        feat1 = np.load(path)
        feat = feat1[feat1.files[0]]
        print(f'Original Feature shape: {feat.shape}')# shape (C, D, H, W) or (C, H, W) or (C, N)
        if feat.ndim > 1:
            #feat_vec = feat.mean(tuple(range(1, feat.ndim)))  # global average
            feat_vec = feat.mean(axis=(2, 3))  # shape: (320, 48)

            # Option 2: Flatten everything except batch
            #features_flat = features.view(320, -1)  # shape: (320, 48*272*256)
            print(f'Averaged Feature shape: {feat_vec.shape}')
        else:
            feat_vec = feat                                  # already 1‑D
        vectors[nn_id] = feat_vec.astype(np.float32)
    return vectors


def combine_features(train_vecs: dict, test_vecs: dict) -> tuple[list[str], np.ndarray]:
    """
    Concatenates train+test dicts → list(ids), 2‑D matrix (N, C).
    Keeps only cases that appear in either set (duplicates resolved by preferring train vec).
    """
    all_ids = list(train_vecs.keys() | test_vecs.keys())
    feats = []
    for k in all_ids:
        v = train_vecs.get(k, test_vecs.get(k))
        feats.append(v)
    return all_ids, np.vstack(feats)  # (N, C)


def main(feature_dir_tr: str, feature_dir_ts: str, csv_path_tr: str, csv_path_ts, image_dir):
    # 1. load subtype table
    df_tr = pd.read_csv(csv_path_tr, dtype=str)
    df_tr.columns = df_tr.columns.str.strip()
    print(f'df_tr columns: {df_tr.columns}')
    df_ts = pd.read_csv(csv_path_ts, dtype=str)
    df_ts.columns = df_ts.columns.str.strip()
    print(f'df_ts columns: {df_ts.columns}')

    df_tr = df_tr[['case_id', 'subtype']].dropna()
    df_ts = df_ts[['case_id', 'subtype']].dropna()

    df = pd.concat([df_tr, df_ts], ignore_index=True)

    print(f'df columns: {df.columns}')
    print(df['subtype'].isnull().sum())  #

    # Normalize column names to remove any invisible characters
    df.columns = df.columns.str.strip()

    subtype_map = dict(zip(df['case_id'], df['subtype']))

    run_pca(feature_dir_tr)

    #
    # # 2. load feature vectors
    # vec_tr = load_feature_vectors(feature_dir_tr)
    # vec_ts = load_feature_vectors(feature_dir_ts)
    #
    # # 3. merge
    # case_ids, X = combine_features(vec_tr, vec_ts)   # X: (N, C)
    #
    # # 4. attach labels (unknowns marked as 'NA')
    # y = [subtype_map.get(cid, 'NA') for cid in case_ids]
    #
    # # 5. UMAP dimensionality reduction
    # #n-components: 2d data
    # #n_neighbours: considers 15 nearest neighbours for each point
    # reducer = umap.UMAP(
    #     n_components=2,
    #     n_neighbors=15,
    #     min_dist=0.1,
    #     metric="cosine",
    #     random_state=42
    # )
    # X2d = reducer.fit_transform(X)  # (N, 2)
    #
    # # 6. color palette
    # unique_labels = sorted(set(y))
    # cmap = plt.cm.tab20
    # color_lookup = {lab: cmap(i % 20) for i, lab in enumerate(unique_labels)}
    #
    # # 7. scatter plot
    # plt.figure(figsize=(8, 6))
    # for lab in unique_labels:
    #     idx = [i for i, l in enumerate(y) if l == lab]
    #     plt.scatter(
    #         X2d[idx, 0], X2d[idx, 1],
    #         s=25, c=[color_lookup[lab]] * len(idx), label=lab, alpha=0.8
    #     )
    # plt.xlabel("UMAP‑1")
    # plt.ylabel("UMAP‑2")
    # plt.title("ROI Feature Map Clusters by Subtype")
    # plt.legend(fontsize=8, loc='best', markerscale=1)
    # plt.tight_layout()
    # plt.savefig(image_dir, dpi=300)
    # plt.show()
    #
    # # 8. print some cluster insight
    # print("\nCases per subtype:")
    # counts = defaultdict(int)
    # for l in y:
    #     counts[l] += 1
    # for lab, n in counts.items():
    #     print(f"  {lab:15s} : {n:4d}")
    #
    # print("\nSaved plot -> umap_subtype_clusters.png")


if __name__ == "__main__":
    tabular_data_dir_tr = r'/gpfs/home6/palfken/masters_thesis/conf_train.csv'

    tabular_data_dir_ts = r'/gpfs/home6/palfken/masters_thesis/conf_val.csv'

    # Feature maps stored i folder always with name patient_id_features.npy
    feature_dir_Tr = sys.argv[0]
    feature_dir_Ts = sys.argv[1]
    image_dir = sys.argv[2]

    main(feature_dir_Tr, feature_dir_Ts, tabular_data_dir_tr, tabular_data_dir_ts,image_dir)
