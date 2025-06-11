import pandas as pd
#
# fold_0 = pd.read_csv(r'/home/bmep/plalfken/my-scratch/nnUNet/fold_0')
# fold_1= pd.read_csv(r'/home/bmep/plalfken/my-scratch/nnUNet/fold_1')
# fold_2 = pd.read_csv(r'/home/bmep/plalfken/my-scratch/nnUNet/fold_2')
# fold_3 = pd.read_csv(r'/home/bmep/plalfken/my-scratch/nnUNet/fold_3')
# fold_4 = pd.read_csv(r'/home/bmep/plalfken/my-scratch/nnUNet/fold_4')
# tabular_data_dir = r'/home/bmep/plalfken/my-scratch/Downloads/WORC_data/WORC_all_with_nnunet_ids.csv'
# tabular_data = pd.read_csv(tabular_data_dir)
# subtype = tabular_data[['nnunet_id','Final_Classification']]



# Load subtype lookup table
tabular_data = pd.read_csv('/home/bmep/plalfken/my-scratch/Downloads/WORC_data/WORC_all_with_nnunet_ids.csv')
subtype_lookup = tabular_data[['nnunet_id', 'Final_Classification']].rename(columns={
    'nnunet_id': 'case_id',
    'Final_Classification': 'subtype'
})

# Paths to fold CSVs
fold_paths = {
    'fold_0': '/home/bmep/plalfken/my-scratch/nnUNet/fold_0',
    'fold_1': '/home/bmep/plalfken/my-scratch/nnUNet/fold_1',
    'fold_2': '/home/bmep/plalfken/my-scratch/nnUNet/fold_2',
    'fold_3': '/home/bmep/plalfken/my-scratch/nnUNet/fold_3',
    'fold_4': '/home/bmep/plalfken/my-scratch/nnUNet/fold_4',
}

# # Add subtype to each fold
# for fold_name, path in fold_paths.items():
#     fold_df = pd.read_csv(path)
#     fold_df = fold_df.merge(subtype_lookup, on='case_id', how='left')  # left join on case_id
#     fold_df.to_csv(path, index=False)  # Overwrite with subtype added
#     print(f"{fold_name} updated with subtype column.")

# Containers for results
overall_fold_scores = {}
class_scores_per_fold = {}

# Process each fold
for fold_name, path in fold_paths.items():
    df = pd.read_csv(path)
    columns = df.columns

    # Drop missing or invalid dice/subtype values
    #df = df.dropna(subset=['dice', 'subtype'])

    # Overall Dice for the fold
    if 'dice' in columns:
        overall_dice = df['dice'].mean()
        overall_fold_scores[fold_name] = overall_dice

        # Dice per subtype (class) for the fold
        class_dice = df.groupby('subtype')['dice'].mean().to_dict()
        class_scores_per_fold[fold_name] = class_dice
    else:
        overall_dice = df['Dice'].mean()
        overall_fold_scores[fold_name] = overall_dice

        # Dice per subtype (class) for the fold
        class_dice = df.groupby('subtype')['Dice'].mean().to_dict()
        class_scores_per_fold[fold_name] = class_dice

# Display results
print("=== Overall Dice per Fold ===")
for fold, score in overall_fold_scores.items():
    print(f"{fold}: {score:.4f}")

print("\n=== Dice per Class in Each Fold ===")
for fold, scores in class_scores_per_fold.items():
    print(f"\n{fold}:")
    for subtype, dice in scores.items():
        print(f"  {subtype}: {dice:.4f}")


from collections import defaultdict

# Collect scores per class
class_aggregates = defaultdict(list)
for fold_dict in class_scores_per_fold.values():
    for subtype, dice in fold_dict.items():
        class_aggregates[subtype].append(dice)

# Compute mean and std
class_stats = {
    cls: {
        "Mean Dice": round(sum(scores) / len(scores), 4),
        "Std Dev": round(pd.Series(scores).std(), 4)
    }
    for cls, scores in class_aggregates.items()
}

# Convert to DataFrame
df_stats = pd.DataFrame.from_dict(class_stats, orient='index')
df_stats["Mean ± Std"] = df_stats["Mean Dice"].astype(str) + " ± " + df_stats["Std Dev"].astype(str)
df_stats = df_stats[["Mean ± Std"]]  # Keep only the final column for LaTeX

print(df_stats)