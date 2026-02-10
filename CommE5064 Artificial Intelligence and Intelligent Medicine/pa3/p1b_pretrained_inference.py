"""
P1(b): Show predicted segmentation using model's default pretrained weights
"""

import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    Orientationd, ScaleIntensityRanged, EnsureTyped, Activations, AsDiscrete
)
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader, decollate_batch

print("=" * 70)
print("P1(b): PRETRAINED MODEL INFERENCE")
print("=" * 70)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define file paths
image_path = 'Workshop3_materials/imagesTr/liver_11.nii.gz'
label_path = 'Workshop3_materials/labelsTr/liver_11.nii.gz'

# Create data dictionary
data_dict = [{"image": image_path, "label": label_path}]

# Define preprocessing transforms
val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(
        keys=["image"], 
        a_min=-175, 
        a_max=250, 
        b_min=0.0, 
        b_max=1.0, 
        clip=True
    ),
    EnsureTyped(keys=["image", "label"]),
])

print("\nPreprocessing data...")
val_ds = Dataset(data=data_dict, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

# Get preprocessed data
batch = next(iter(val_loader))
image = batch["image"].to(device)
label = batch["label"].to(device)

print(f"Preprocessed image shape: {image.shape}")
print(f"Preprocessed label shape: {label.shape}")

# Initialize SegResNet model
print("\nInitializing SegResNet model...")
model = SegResNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,  # background, liver, tumor
    init_filters=32,
    dropout_prob=0.2,
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Try to load pretrained weights from MONAI model zoo
print("\nAttempting to load pretrained weights...")
try:
    from monai.bundle import download
    # Try to download pretrained weights
    # Note: This might fail due to network restrictions
    pretrained_path = download(name="segresnet_liver", source="monai")
    model.load_state_dict(torch.load(pretrained_path))
    print("✓ Pretrained weights loaded successfully")
    weights_loaded = True
except Exception as e:
    print(f"✗ Could not load pretrained weights: {e}")
    print("  Using randomly initialized weights instead")
    weights_loaded = False

# Run inference
print("\nRunning inference with sliding window...")
model.eval()
with torch.no_grad():
    # Use sliding window inference for large volumes
    roi_size = (96, 96, 96)  # Patch size for inference
    sw_batch_size = 4
    
    pred = sliding_window_inference(
        inputs=image,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=0.5
    )
    
    # Apply softmax and get argmax
    pred_softmax = torch.softmax(pred, dim=1)
    pred_label = torch.argmax(pred_softmax, dim=1)

print(f"Prediction shape: {pred_label.shape}")

# Move back to CPU for visualization
pred_np = pred_label.cpu().numpy()[0]
label_np = label.cpu().numpy()[0, 0]
image_np = image.cpu().numpy()[0, 0]

print(f"\nPrediction unique values: {np.unique(pred_np)}")
print(f"Ground truth unique values: {np.unique(label_np)}")

# Calculate Dice coefficient for liver (class 1)
def dice_coefficient(pred, target, class_id=1):
    pred_mask = (pred == class_id)
    target_mask = (target == class_id)
    intersection = np.sum(pred_mask & target_mask)
    union = np.sum(pred_mask) + np.sum(target_mask)
    if union == 0:
        return 0.0
    return 2.0 * intersection / union

dice_liver = dice_coefficient(pred_np, label_np, class_id=1)
print(f"\nDice coefficient (Liver - class 1): {dice_liver:.4f}")

if 2 in np.unique(label_np):
    dice_tumor = dice_coefficient(pred_np, label_np, class_id=2)
    print(f"Dice coefficient (Tumor - class 2): {dice_tumor:.4f}")

# Visualize predictions at multiple slices
print("\nCreating visualization...")
slices_to_show = [
    pred_np.shape[2] // 4,      # 25% depth
    pred_np.shape[2] // 2,      # 50% depth (middle)
    3 * pred_np.shape[2] // 4   # 75% depth
]

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for i, slice_idx in enumerate(slices_to_show):
    # Original image
    axes[i, 0].imshow(image_np[:, :, slice_idx], cmap='gray')
    axes[i, 0].set_title(f'Input CT (Slice {slice_idx})')
    axes[i, 0].axis('off')
    
    # Ground truth
    axes[i, 1].imshow(image_np[:, :, slice_idx], cmap='gray')
    axes[i, 1].imshow(label_np[:, :, slice_idx], cmap='jet', alpha=0.4)
    axes[i, 1].set_title('Ground Truth Overlay')
    axes[i, 1].axis('off')
    
    # Prediction
    axes[i, 2].imshow(image_np[:, :, slice_idx], cmap='gray')
    axes[i, 2].imshow(pred_np[:, :, slice_idx], cmap='jet', alpha=0.4)
    axes[i, 2].set_title('Prediction Overlay')
    axes[i, 2].axis('off')
    
    # Side by side comparison
    axes[i, 3].imshow(label_np[:, :, slice_idx], cmap='jet')
    axes[i, 3].imshow(pred_np[:, :, slice_idx], cmap='jet', alpha=0.5)
    axes[i, 3].set_title('GT (solid) vs Pred (transparent)')
    axes[i, 3].axis('off')

plt.suptitle(f'P1(b): Pretrained SegResNet Predictions (Dice={dice_liver:.4f})', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('p1b_pretrained_predictions.png', dpi=150, bbox_inches='tight')
print("Visualization saved to: p1b_pretrained_predictions.png")

# Save statistics
with open('p1b_results.txt', 'w') as f:
    f.write("P1(b) PRETRAINED MODEL RESULTS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Pretrained weights loaded: {weights_loaded}\n")
    f.write(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    f.write(f"Input shape: {image.shape}\n")
    f.write(f"Output shape: {pred_label.shape}\n")
    f.write(f"Prediction classes: {np.unique(pred_np)}\n")
    f.write(f"\nDice coefficient (Liver): {dice_liver:.4f}\n")
    if 2 in np.unique(label_np):
        f.write(f"Dice coefficient (Tumor): {dice_tumor:.4f}\n")

print("\nResults saved to: p1b_results.txt")
print("=" * 70)