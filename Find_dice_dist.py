import numpy
import pandas as pd

dir_5 ="/gpfs/home6/palfken/Dice_scores_5epochs.csv"
dir_20 = "/gpfs/home6/palfken/Dice_scores_20epochs.csv"


df_5 = pd.read_csv(dir_5)
df_20 = pd.read_csv(dir_20)

df_20 = df_20.rename(columns={"dice_5": "dice_20"})

merged_df = pd.merge(df_5, df_20, on="case_id")

# Save or inspect
merged_df.to_csv("/gpfs/home6/palfken/Dice_scores_all.csv", index=False)
print(merged_df.head())

