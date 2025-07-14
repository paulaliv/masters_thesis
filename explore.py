import pandas as pd
from sklearn.model_selection import train_test_split

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset,nnUNetDatasetBlosc2
tabular_data_all = pd.read_csv('/home/bmep/plalfken/my-scratch/Downloads/WORC_data/WORC_all_with_nnunet_ids.csv')
test_data = pd.read_csv('/home/bmep/plalfken/my-scratch/Downloads/WORC_data/WORC_test.csv')
train_data = pd.read_csv('/home/bmep/plalfken/my-scratch/Downloads/WORC_data/WORC_train.csv')
test_preprocessed = r'/home/bmep/plalfken/my-scratch/STT_classification/Segmentation/nnUNetFrame/nnunet_preprocessed/Dataset002_SoftTissue/nnUNetPlans_3d_fullres'
# print(len(train_data))
# print(len(test_data))
# print(len(tabular_data_all))
#
# print(test_data.columns)
# print('TEST DATA DISTRIBUTIOn')
# print(test_data['Final_Classification'].value_counts())
#
# print('TRAIN DATA DISTRIBUTION')
# print(train_data['Final_Classification'].value_counts())
#
# from sklearn.model_selection import StratifiedKFold
# X = train_data.drop(columns=['Final_Classification'])
# y = train_data['Final_Classification']
# skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
# for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
#     train_fold = y.iloc[train_idx]
#     test_fold = y.iloc[test_idx]
#
#     train_counts = train_fold.value_counts().sort_index()
#     test_counts = test_fold.value_counts().sort_index()
#
#
#
#     print(f'Fold {fold +1}')
#     print('Train Counts')
#     print(train_counts)
#     print('Test Counts')
#     print(test_counts)
import pandas as pd

# Load your CSV
df = pd.read_csv("/gpfs/home6/palfken/masters_thesis/tumor_bounding_box_sizes.csv")

# Compute tumor volume in mm³
df["volume_mm3"] = df["bbox_mm_x"] * df["bbox_mm_y"] * df["bbox_mm_z"]

# Global stats
global_stats = df["volume_mm3"].agg(["min", "max", "mean", "std"])
print("Global Tumor Volume Stats (mm³):\n", global_stats)

# Stats per tumor subtype
per_subtype_stats = df.groupby("tumor_class")["volume_mm3"].agg(["min", "max", "mean", "std"])
print("\nPer-Subtype Tumor Volume Stats (mm³):\n", per_subtype_stats)
