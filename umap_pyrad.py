import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

# --- Load and preprocess data ---
data_path = "/gpfs/home6/palfken/radiomics_features.csv"
df = pd.read_csv(data_path)

# Map tumor_class to priority
priority_mapping = {
    'LeiomyoSarcomas': 'high_malignant',
    'MyxoidlipoSarcoma': 'moderate_malignant',
    'MyxofibroSarcomas': 'moderate_malignant',
    'WDLPS': 'low_malignant',
    'DTF': 'intermediate'
}
df['priority'] = df['tumor_class'].map(priority_mapping)

# --- Feature standardization ---
features = df.drop(columns=['case_id', 'tumor_class', 'priority'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# --- PCA + UMAP ---
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_pca)

# --- Define custom consistent colors per priority ---

tumor_classes = df['tumor_class'].unique()
tumor_class_palette = dict(zip(tumor_classes, sns.color_palette('tab20', len(tumor_classes))))

# Custom priority palette (single color for moderate_malignant)
priority_order = ['low_malignant', 'intermediate', 'moderate_malignant', 'high_malignant']
priority_palette = dict(zip(priority_order, sns.color_palette('Set2', len(priority_order))))


# --- Plot side-by-side ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot UMAP colored by tumor_class (using mapped priority colors)
sns.scatterplot(
    x=X_umap[:, 0], y=X_umap[:, 1],
    hue=df['tumor_class'],
    palette=tumor_class_palette,
    s=60, edgecolor='k', ax=axes[0]
)
axes[0].set_title("UMAP colored by Tumor Class (PCA 95%)")
axes[0].set_xlabel("UMAP-1")
axes[0].set_ylabel("UMAP-2")
axes[0].legend(title="Tumor Class", bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot UMAP colored by priority (with separate but consistent palette)
sns.scatterplot(
    x=X_umap[:, 0], y=X_umap[:, 1],
    hue=df['priority'],
    palette=priority_palette,
    s=60, edgecolor='k', ax=axes[1]
)
axes[1].set_title("UMAP colored by Priority Group (PCA 95%)")
axes[1].set_xlabel("UMAP-1")
axes[1].set_ylabel("UMAP-2")
axes[1].legend(title="Priority Group", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
