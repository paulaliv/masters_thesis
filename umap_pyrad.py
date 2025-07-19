import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import numpy as np

# --- Load Data ---
data_path = "/gpfs/home6/palfken/radiomics_features.csv"
df = pd.read_csv(data_path)

# Extract tumor_class
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

# --- Feature preprocessing ---
features = df.drop(columns=['case_id', 'tumor_class', 'priority'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# --- PCA: retain 95% variance ---
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA retained components: {X_pca.shape[1]} for 95% variance")

# --- UMAP ---
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_pca)

# --- Set up side-by-side plots ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Define consistent colors for each tumor class
unique_classes = tumor_class.unique()
palette = sns.color_palette('Set2', len(unique_classes))
class_color_dict = dict(zip(unique_classes, palette))

# Map tumor_class to consistent colors
sns.scatterplot(
    x=X_umap[:, 0], y=X_umap[:, 1],
    hue=tumor_class,
    palette=class_color_dict,
    s=60, edgecolor='k', ax=axes[0]
)
axes[0].set_title("UMAP colored by Tumor Class (PCA 95%)")
axes[0].set_xlabel("UMAP-1")
axes[0].set_ylabel("UMAP-2")
axes[0].legend(title="Tumor Class", bbox_to_anchor=(1.05, 1), loc='upper left')

# Merge tumor_class colors into priority (will merge two classes into one)
priority_palette = {
    priority: class_color_dict[tumor]
    for tumor, priority in priority_mapping.items()
    if priority_mapping.values()
}
# However, some priorities share tumor_class → average or re-assign color
# For simplicity, use a distinct palette per priority instead
priority_palette = dict(zip(df['priority'].unique(), sns.color_palette('Set1', len(df['priority'].unique()))))

# Plot priority-colored UMAP
sns.scatterplot(
    x=X_umap[:, 0], y=X_umap[:, 1],
    hue=df['priority'],
    palette=priority_palette,
    s=60, edgecolor='k', ax=axes[1]
)
axes[1].set_title("UMAP colored by Priority Group (PCA 95%)")
axes[1].set_xlabel("UMAP-1")
axes[1].set_ylabel("UMAP-2")
axes[1].legend(title="Priority", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
