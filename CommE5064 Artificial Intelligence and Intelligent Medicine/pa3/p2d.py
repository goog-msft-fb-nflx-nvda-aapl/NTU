"""
P2(d): Build Xtrain and Xtest (max length = 24, zero-padding as needed)
"""

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

print("=" * 70)
print("P2(d): BUILDING Xtrain AND Xtest")
print("=" * 70)

# Load imputed sequences
with open('p2c_train_imputed.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('p2c_test_imputed.pkl', 'rb') as f:
    test_data = pickle.load(f)

print(f"Loaded {len(train_data['sequences'])} training patients")
print(f"Loaded {len(test_data['sequences'])} test patients")

# Define feature parameters (exclude Hours and RecordID)
STATIC_PARAMS = ['Age', 'Gender', 'Height', 'ICUType', 'Weight']
TIME_VARYING_PARAMS = [
    'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol',
    'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT',
    'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent', 'Na', 'NIDiasABP',
    'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
    'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
]

FEATURE_PARAMS = STATIC_PARAMS + TIME_VARYING_PARAMS
NUM_FEATURES = len(FEATURE_PARAMS)
MAX_LENGTH = 24

print(f"\n{'='*70}")
print("DATASET CONFIGURATION")
print(f"{'='*70}")
print(f"Number of features per time step: {NUM_FEATURES}")
print(f"Feature parameters: {FEATURE_PARAMS[:5]}... ({NUM_FEATURES} total)")
print(f"Max sequence length: {MAX_LENGTH}")
print(f"Padding strategy: Zero-padding (prepend zeros)")

def sequence_to_array(sequence, feature_params, max_length):
    """
    Convert a patient sequence to a fixed-length array
    
    Args:
        sequence: List of time point dictionaries
        feature_params: List of parameter names to extract
        max_length: Maximum sequence length (pad/truncate to this)
    
    Returns:
        numpy array of shape (max_length, num_features)
    """
    seq_length = len(sequence)
    num_features = len(feature_params)
    
    # Initialize with zeros
    array = np.zeros((max_length, num_features), dtype=np.float32)
    
    if seq_length == 0:
        # Empty sequence, return all zeros
        return array
    
    # Extract features for each time point
    sequence_array = np.zeros((seq_length, num_features), dtype=np.float32)
    for i, time_point in enumerate(sequence):
        for j, param in enumerate(feature_params):
            value = time_point.get(param, 0.0)
            sequence_array[i, j] = float(value) if pd.notna(value) else 0.0
    
    if seq_length <= max_length:
        # Sequence is shorter than max_length: zero-pad at the beginning
        # This is important for RNNs to focus on recent measurements
        array[-seq_length:, :] = sequence_array
    else:
        # Sequence is longer than max_length: take the last max_length time points
        array = sequence_array[-max_length:, :]
    
    return array

print(f"\n{'='*70}")
print("BUILDING Xtrain")
print(f"{'='*70}")

Xtrain = []
for sequence in tqdm(train_data['sequences'], desc="Processing train"):
    array = sequence_to_array(sequence, FEATURE_PARAMS, MAX_LENGTH)
    Xtrain.append(array)

Xtrain = np.array(Xtrain)

print(f"\n{'='*70}")
print("BUILDING Xtest")
print(f"{'='*70}")

Xtest = []
for sequence in tqdm(test_data['sequences'], desc="Processing test"):
    array = sequence_to_array(sequence, FEATURE_PARAMS, MAX_LENGTH)
    Xtest.append(array)

Xtest = np.array(Xtest)

print(f"\n{'='*70}")
print("DATASET SHAPES")
print(f"{'='*70}")
print(f"Xtrain shape: {Xtrain.shape}")
print(f"  - {Xtrain.shape[0]} patients")
print(f"  - {Xtrain.shape[1]} time steps (max)")
print(f"  - {Xtrain.shape[2]} features")
print(f"\nXtest shape: {Xtest.shape}")
print(f"  - {Xtest.shape[0]} patients")
print(f"  - {Xtest.shape[1]} time steps (max)")
print(f"  - {Xtest.shape[2]} features")

print(f"\n{'='*70}")
print("DATA STATISTICS")
print(f"{'='*70}")
print(f"Xtrain statistics:")
print(f"  Mean: {Xtrain.mean():.4f}")
print(f"  Std: {Xtrain.std():.4f}")
print(f"  Min: {Xtrain.min():.4f}")
print(f"  Max: {Xtrain.max():.4f}")
print(f"  Non-zero ratio: {(Xtrain != 0).sum() / Xtrain.size:.4f}")

print(f"\nXtest statistics:")
print(f"  Mean: {Xtest.mean():.4f}")
print(f"  Std: {Xtest.std():.4f}")
print(f"  Min: {Xtest.min():.4f}")
print(f"  Max: {Xtest.max():.4f}")
print(f"  Non-zero ratio: {(Xtest != 0).sum() / Xtest.size:.4f}")

print(f"\n{'='*70}")
print("EXAMPLE: First patient in training set")
print(f"{'='*70}")
sample_idx = 0
sample_id = train_data['patient_ids'][sample_idx]
sample_seq_len = train_data['sequence_lengths'][sample_idx]

print(f"Patient ID: {sample_id}")
print(f"Original sequence length: {sample_seq_len}")
print(f"Array shape: {Xtrain[sample_idx].shape}")

# Show padding
padding_rows = MAX_LENGTH - sample_seq_len
print(f"\nZero-padding:")
print(f"  Padded rows (beginning): {padding_rows}")
print(f"  Data rows (end): {sample_seq_len}")

# Show first few rows (should be zeros if padded)
print(f"\nFirst 3 rows (should be zeros if sequence < {MAX_LENGTH}):")
print(Xtrain[sample_idx][:3, :5])  # First 3 timesteps, first 5 features

# Show last few rows (should be actual data)
print(f"\nLast 3 rows (actual data):")
print(Xtrain[sample_idx][-3:, :5])  # Last 3 timesteps, first 5 features

print(f"\nFeature names for first 5 columns:")
for i, param in enumerate(FEATURE_PARAMS[:5]):
    print(f"  Column {i}: {param}")

# Save arrays
print(f"\n{'='*70}")
print("SAVING ARRAYS")
print(f"{'='*70}")

np.save('p2d_Xtrain.npy', Xtrain)
print(f"✓ Saved: p2d_Xtrain.npy")

np.save('p2d_Xtest.npy', Xtest)
print(f"✓ Saved: p2d_Xtest.npy")

# Also save feature names and metadata
metadata = {
    'feature_params': FEATURE_PARAMS,
    'num_features': NUM_FEATURES,
    'max_length': MAX_LENGTH,
    'train_patient_ids': train_data['patient_ids'],
    'test_patient_ids': test_data['patient_ids'],
    'train_sequence_lengths': train_data['sequence_lengths'],
    'test_sequence_lengths': test_data['sequence_lengths']
}

with open('p2d_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print(f"✓ Saved: p2d_metadata.pkl")

print(f"\n{'='*70}")
print("P2(d) COMPLETE!")
print(f"{'='*70}")
print("\nSummary:")
print(f"  ✓ Xtrain: {Xtrain.shape}")
print(f"  ✓ Xtest: {Xtest.shape}")
print(f"  ✓ Features per timestep: {NUM_FEATURES}")
print(f"  ✓ Max sequence length: {MAX_LENGTH}")
print(f"  ✓ Padding: Zero-padding at beginning")