import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load the CT image and label
image_path = 'Workshop3_materials/imagesTr/liver_11.nii.gz'
label_path = 'Workshop3_materials/labelsTr/liver_11.nii.gz'

# Load data
image = nib.load(image_path)
label = nib.load(label_path)

# Get data arrays
image_data = image.get_fdata()
label_data = label.get_fdata()

print("=" * 60)
print("CT IMAGE EXPLORATION")
print("=" * 60)
print(f"Image shape: {image_data.shape}")
print(f"Image dtype: {image_data.dtype}")
print(f"Image min/max: {image_data.min():.2f} / {image_data.max():.2f}")
print(f"Image mean/std: {image_data.mean():.2f} / {image_data.std():.2f}")
print(f"\nLabel shape: {label_data.shape}")
print(f"Label unique values: {np.unique(label_data)}")
print(f"Label dtype: {label_data.dtype}")

# Count liver pixels
liver_pixels = np.sum(label_data > 0)
total_pixels = np.prod(label_data.shape)
print(f"\nLiver pixels: {liver_pixels} ({100*liver_pixels/total_pixels:.2f}%)")
print(f"Background pixels: {total_pixels - liver_pixels} ({100*(total_pixels-liver_pixels)/total_pixels:.2f}%)")

# Visualize middle slice
mid_slice = image_data.shape[2] // 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image_data[:, :, mid_slice], cmap='gray')
axes[0].set_title(f'CT Image - Slice {mid_slice}')
axes[0].axis('off')

axes[1].imshow(label_data[:, :, mid_slice], cmap='jet')
axes[1].set_title(f'Ground Truth Label - Slice {mid_slice}')
axes[1].axis('off')

axes[2].imshow(image_data[:, :, mid_slice], cmap='gray')
axes[2].imshow(label_data[:, :, mid_slice], cmap='jet', alpha=0.3)
axes[2].set_title(f'Overlay - Slice {mid_slice}')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('data_exploration.png', dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: data_exploration.png")
print("=" * 60)