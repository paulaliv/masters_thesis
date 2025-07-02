import os
import pandas as pd



fold_paths = {
    'fold_0': '/home/bmep/plalfken/my-scratch/nnUNet/fold_0',
    'fold_1': '/home/bmep/plalfken/my-scratch/nnUNet/fold_1',
    'fold_2': '/home/bmep/plalfken/my-scratch/nnUNet/fold_2',
    'fold_3': '/home/bmep/plalfken/my-scratch/nnUNet/fold_3',
    'fold_4': '/home/bmep/plalfken/my-scratch/nnUNet/fold_4',
}
for fold in range(5):
    fold_name = f'fold_{fold}'
    fold_file = fold_paths[fold_name]

    # Load the CSV
    metadata = pd.read_csv(fold_file)

    # Clean up whitespaces just in case
    metadata['subtype'] = metadata['subtype'].str.strip()

    # Add the merged subtype column
    metadata['subtype_merged'] = metadata['subtype'].replace({
        'WDLPS': 'LipoSarcoma',
        'MyxoidlipoSarcoma': 'LipoSarcoma'
    })

    # Overwrite the file
    metadata.to_csv(fold_file, index=False)

    print(f"{fold_file} updated.")
