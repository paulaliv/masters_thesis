import pandas as pd
import glob
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import seaborn as sns
# path to your CSVs


def plot_confusion(y_true, y_pred, title, save_path=None):
    class_names = ["Fail (0-0.1)", "Poor (0.1-0.5)", "Moderate (0.5-0.7)", "Good (>0.7)"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

csv_files = glob.glob("/home/bmep/plalfken/my-scratch/results/*_UMC_results_mask_ordinal.csv")

results = []
metrics = ['confidence', 'entropy','mutual_info','epkl']
pretty_names = {
    "confidence": "Confidence",
    "entropy": "Entropy",
    "mutual_info": "Mutual Info",
    "epkl": "EPKL"
}
for f in csv_files:

    metric_found = None
    for m in metrics:
        if m in f.lower():  # case-insensitive match
            metric_found = pretty_names[m]
            break

    if metric_found is None:
        print(f"⚠️ No metric found in filename: {f}")
        continue

    print(f"  → Metric: {metric_found}")

    df = pd.read_csv(f)
    print(df.columns)
    y_true = df["gt"]
    y_pred = df["maj_pred"]

    kappa = cohen_kappa_score(y_true, y_pred, weights=None)
    quad_kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    results.append({
        "file": f.replace("/home/bmep/plalfken/my-scratch/results","" ),
        "loss": "Ordinal",
        "modality": "Mask",
        "map": metric_found,
        "set": "UMC",
        "kappa": kappa,
        "quadratic_kappa": quad_kappa
    })
    save_path = os.path.join(
        "/home/bmep/plalfken/my-scratch/results",f"heatmap_mask_{metric_found}_ordinal_UMC.png"

    )

    plot_confusion(y_true, y_pred,
                   title=f"Model with {metric_found} and Mask \n on UMC Set (K$^2$ = {quad_kappa:.3f})",
                   save_path=save_path)

results_df = pd.DataFrame(results)
print(results_df)

csv_dir =("/home/bmep/plalfken/my-scratch/results/Test_results_all.csv")
old_df = pd.read_csv(csv_dir)

key_cols = ["file", "loss", "modality", "set"]  # columns to check for duplicates

# Keep only rows not already in old_df based on key columns
new_rows = results_df[~results_df.set_index(key_cols).index.isin(old_df.set_index(key_cols).index)]



print(f"Adding {len(new_rows)} new rows.")

# Concatenate
combined_df = pd.concat([old_df, new_rows], ignore_index=True)


print(f"Total rows after combining: {len(combined_df)}")

# Save the combined DataFrame back to the same CSV
combined_df.to_csv(csv_dir, index=False)
