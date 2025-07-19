import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# --- Load Data ---
# Replace this with the path to your radiomics features file
data_path = "/gpfs/home6/palfken/radiomics_features.csv"
df = pd.read_csv(data_path)


# Extract tumor_class separately
tumor_class = df['tumor_class']

# Add priority group
priority_mapping = {
    'LeiomyoSarcomas': 'high_malignant',
    'MyxoidlipoSarcoma': 'moderate_malignant',
    'MyxofibroSarcomas': 'moderate_malignant',
    'WDLPS': 'low_malignant',
    'DTF': 'intermediate'
}
df['priority'] = df['tumor_class'].map(priority_mapping)

# Extract features (drop all non-feature columns)
features = df.drop(columns=['tumor_class', 'priority'])

# --- Standardize features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

variance_levels = [0.90, 0.93, 0.95]

# --- Set up plots ---
fig, axes = plt.subplots(1, len(variance_levels), figsize=(18, 6))

for i, var_threshold in enumerate(variance_levels):
    # PCA to retain specific variance
    pca = PCA(n_components=var_threshold)
    X_pca = pca.fit_transform(X_scaled)

    # Print number of components retained
    print(f"Variance threshold: {var_threshold * 100:.0f}%, PCA components retained: {X_pca.shape[1]}")

    # UMAP to 2D
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_pca)

    # Plot
    sns.scatterplot(
        x=X_umap[:, 0], y=X_umap[:, 1],
        hue=tumor_class,
        palette='Set2',
        s=60,
        edgecolor='k',
        ax=axes[i]
    )
    axes[i].set_title(f"UMAP (PCA {int(var_threshold * 100)}% var, {X_pca.shape[1]} comps)")
    axes[i].set_xlabel("UMAP-1")
    axes[i].set_ylabel("UMAP-2")

plt.tight_layout()
plt.legend(title='Tumor Class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


fig, axes = plt.subplots(1, len(variance_levels), figsize=(18, 6))

for i, var_threshold in enumerate(variance_levels):
    # PCA to retain specific variance
    pca = PCA(n_components=var_threshold)
    X_pca = pca.fit_transform(X_scaled)

    # Print number of components retained
    print(f"Variance threshold: {var_threshold * 100:.0f}%, PCA components retained: {X_pca.shape[1]}")

    # UMAP to 2D
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_pca)

    # Plot
    sns.scatterplot(
        x=X_umap[:, 0], y=X_umap[:, 1],
        hue=df['priority'],
        palette='Set2',
        s=60,
        edgecolor='k',
        ax=axes[i]
    )
    axes[i].set_title(f"UMAP (PCA {int(var_threshold * 100)}% var, {X_pca.shape[1]} comps)")
    axes[i].set_xlabel("UMAP-1")
    axes[i].set_ylabel("UMAP-2")

plt.tight_layout()
plt.legend(title="Priority Group", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()