import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import os
mask_dir = f''
feature_map_dir = f''

roi_features = np.load(feature_map_dir)       # shape: (C, Z, Y, X)
cropped_mask = np.load(mask_dir)      # shape: (D, H, W)

print("Feature shape:", roi_features.shape)
print("Mask shape:", cropped_mask.shape)

# --- Resize mask to match feature map ---
_, Z, Y, X = roi_features.shape

resized_mask = resize(
    cropped_mask,
    output_shape=(Z, Y, X),
    order=0,                    # Nearest-neighbor for label preservation
    preserve_range=True,
    anti_aliasing=False
).astype(np.uint8)

# --- Visualization ---
feature_channel = 0
num_slices = 5
slice_indices = np.linspace(0, Z - 1, num=num_slices, dtype=int)

fig, axs = plt.subplots(1, num_slices, figsize=(16, 5))

for i, z in enumerate(slice_indices):
    feature_slice = roi_features[feature_channel, z, :, :]
    mask_slice = resized_mask[z, :, :]

    ax = axs[i]
    ax.imshow(feature_slice, cmap='gray')
    if np.any(mask_slice):
        ax.imshow(mask_slice, cmap='Reds', alpha=0.4)
    ax.set_title(f'Z={z}')
    ax.axis('off')

plt.tight_layout()
save_path = '/path/to/save/feature_mask_overlay.png'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, bbox_inches='tight')
plt.close()

print(f"Visualization saved to {save_path}")