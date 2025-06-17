import numpy as np
from scipy.spatial import distance
from sympy.solvers.diophantine.diophantine import prime_as_sum_of_two_squares
import pandas as pd
# data containing patient id under nnunet_id and label under Final_Classification
tabular_data_dir = r'/home/bmep/plalfken/my-scratch/Downloads/WORC_data/WORC_all_with_nnunet_ids.csv'
tabular_data = pd.read_csv(tabular_data_dir)
# Feature maps stored i folder always with name patient_id_features.npy
feature_dir = r'/home/bmep/plalfken/my-scratch/Downloads'

def flatten(feature_map):
    pass


def group_by_class(features, labels):
    pass

def compute_stats(flattened_features):
    mean_vec = np.mean(flattened_features, axis=0)  # shape (D,)
    cov_matrix = np.cov(flattened_features, rowvar=False)  # shape (D, D)

    # Step 2: Invert covariance matrix
    inv_cov_matrix = np.linalg.inv(cov_matrix)


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
