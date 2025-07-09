
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
fold_paths = {
    'fold_0': '/home/bmep/plalfken/my-scratch/nnUNet/fold_0',
    'fold_1': '/home/bmep/plalfken/my-scratch/nnUNet/fold_1',
    'fold_2': '/home/bmep/plalfken/my-scratch/nnUNet/fold_2',
    'fold_3': '/home/bmep/plalfken/my-scratch/nnUNet/fold_3',
    'fold_4': '/home/bmep/plalfken/my-scratch/nnUNet/fold_4',
}

# Collect all dice scores
all_dice_scores = []

for fold in range(5):
        fold_key = f'fold_{fold}'
        df = pd.read_csv(fold_paths[fold_key])
        if 'dice' in df.columns:
            print(f'Number of samples in fold {fold}: {len(df)}')
            all_dice_scores.extend(df['dice'].dropna().tolist())

# Convert to DataFrame for ease of use
dice_df = pd.DataFrame(all_dice_scores, columns=['dice'])

# Define bins and compute bin indices
bin_edges = np.linspace(0, 1, 11)  # 10 bins => 11 edges from 0.0 to 1.0
bin_indices = np.digitize(all_dice_scores, bin_edges, right=False) - 1

# Clip to make sure highest value (==1.0) falls into bin 9
bin_indices = np.clip(bin_indices, 0, 9)

# Count the number of scores per bin
bin_counts = pd.Series(bin_indices).value_counts().sort_index()


# Define bin edges for the 5 bins (Fail, Poor, Moderate, Good, Excellent)
bin_edges = [0.0, 0.1, 0.7, 0.8, 0.90, 0.95, 1.0]

# Use np.digitize to assign bins
bin_indices = np.digitize(all_dice_scores, bin_edges, right=False) - 1

# Count samples per bin
bin_counts = np.bincount(bin_indices, minlength=len(bin_edges)-1)

# Print counts with bin labels
bin_labels = [
    '< 0.1',
    '[0.1, 0.70)',
    '[0.70, 0.8)',
    '[0.8, 0.90)',
    '[0.9, 0.95)',
    '≥ 0.95'
]

for label, count in zip(bin_labels, bin_counts):
    print(f'{label}: {count} samples')
# Bin the dice scores
bin_indices = np.digitize(all_dice_scores, bin_edges, right=False) - 1
bin_counts = np.bincount(bin_indices, minlength=len(bin_labels))

# Plot
plt.figure(figsize=(8, 5))
plt.bar(bin_labels, bin_counts, color='skyblue', edgecolor='black')
plt.title('Dice Score Distribution (Custom Bins)')
plt.ylabel('Number of Samples')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()