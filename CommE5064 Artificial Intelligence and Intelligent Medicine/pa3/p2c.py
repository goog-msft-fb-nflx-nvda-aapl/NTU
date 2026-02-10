"""
P2(c): Impute missing values for each patient
Strategy: Forward-fill + median imputation
"""

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

print("=" * 70)
print("P2(c): IMPUTING MISSING VALUES")
print("=" * 70)

# Load sequences
with open('p2b_train_sequences.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('p2b_test_sequences.pkl', 'rb') as f:
    test_data = pickle.load(f)

print(f"Loaded {len(train_data['sequences'])} training patients")
print(f"Loaded {len(test_data['sequences'])} test patients")

# Define parameter groups
STATIC_PARAMS = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']
TIME_VARYING_PARAMS = [
    'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol',
    'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT',
    'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent', 'Na', 'NIDiasABP',
    'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
    'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
]

ALL_PARAMS = STATIC_PARAMS + TIME_VARYING_PARAMS

print(f"\n{'='*70}")
print("COMPUTING GLOBAL STATISTICS FOR IMPUTATION")
print(f"{'='*70}")

# Compute global medians from training set for fallback imputation
def compute_global_medians(sequences):
    """Compute median for each parameter across all patients and time points"""
    param_values = {param: [] for param in ALL_PARAMS}
    
    for sequence in sequences:
        for time_point in sequence:
            for param in ALL_PARAMS:
                value = time_point.get(param, np.nan)
                if pd.notna(value):
                    param_values[param].append(value)
    
    medians = {}
    for param, values in param_values.items():
        if len(values) > 0:
            medians[param] = np.median(values)
        else:
            medians[param] = 0.0  # Fallback if parameter never observed
    
    return medians

print("Computing global medians from training set...")
global_medians = compute_global_medians(train_data['sequences'])

print(f"\nGlobal medians computed for {len(global_medians)} parameters")
print("Sample medians:")
for param in ['Age', 'HR', 'Temp', 'Glucose', 'GCS']:
    if param in global_medians:
        print(f"  {param}: {global_medians[param]:.2f}")

print(f"\n{'='*70}")
print("IMPUTATION STRATEGY")
print(f"{'='*70}")
print("""
1. Static parameters (Age, Gender, Height, etc.):
   - Forward-fill from first occurrence
   - If never observed: use global median

2. Time-varying parameters (HR, BP, labs, etc.):
   - Forward-fill from last observed value (carry forward)
   - If never observed before current time: use global median

3. Hours: Keep original (no imputation needed)
""")

def impute_patient_sequence(sequence, global_medians):
    """
    Impute missing values for a single patient's sequence
    
    Strategy:
    - Static params: forward-fill from first occurrence, then global median
    - Time-varying params: forward-fill (carry last value), then global median
    """
    imputed_sequence = []
    
    # Track last observed values for forward-fill
    last_observed = {}
    
    for time_point in sequence:
        imputed_point = time_point.copy()
        
        # Process each parameter
        for param in ALL_PARAMS:
            current_value = time_point.get(param, np.nan)
            
            if pd.notna(current_value):
                # Value is present, update last observed
                last_observed[param] = current_value
                imputed_point[param] = current_value
            else:
                # Value is missing, try to impute
                if param in last_observed:
                    # Forward-fill from last observed value
                    imputed_point[param] = last_observed[param]
                else:
                    # Never observed before, use global median
                    imputed_point[param] = global_medians[param]
        
        imputed_sequence.append(imputed_point)
    
    return imputed_sequence

# Impute training set
print(f"\n{'='*70}")
print("IMPUTING TRAINING SET")
print(f"{'='*70}")

train_imputed_sequences = []
for sequence in tqdm(train_data['sequences'], desc="Imputing train"):
    imputed_seq = impute_patient_sequence(sequence, global_medians)
    train_imputed_sequences.append(imputed_seq)

# Impute test set
print(f"\n{'='*70}")
print("IMPUTING TEST SET")
print(f"{'='*70}")

test_imputed_sequences = []
for sequence in tqdm(test_data['sequences'], desc="Imputing test"):
    imputed_seq = impute_patient_sequence(sequence, global_medians)
    test_imputed_sequences.append(imputed_seq)

print(f"\n{'='*70}")
print("VERIFICATION - SAME PATIENT BEFORE AND AFTER IMPUTATION")
print(f"{'='*70}")

# Show same patient (first one) before and after imputation
sample_idx = 0
sample_id = train_data['patient_ids'][sample_idx]

print(f"\nPatient ID: {sample_id}")
print(f"Number of time points: {len(train_data['sequences'][sample_idx])}")

print(f"\n{'='*70}")
print("BEFORE IMPUTATION (first 3 time points)")
print(f"{'='*70}")

for i in range(min(3, len(train_data['sequences'][sample_idx]))):
    time_point = train_data['sequences'][sample_idx][i]
    print(f"\nTime point {i+1} (Hour {time_point.get('Hours', 'N/A')}):")
    
    non_nan = sum(1 for k, v in time_point.items() 
                  if k != 'Hours' and pd.notna(v))
    print(f"  Non-missing: {non_nan}/41")
    
    # Show some key parameters
    print(f"  Age: {time_point.get('Age', 'NaN')}")
    print(f"  HR: {time_point.get('HR', 'NaN')}")
    print(f"  Temp: {time_point.get('Temp', 'NaN')}")
    print(f"  Glucose: {time_point.get('Glucose', 'NaN')}")

print(f"\n{'='*70}")
print("AFTER IMPUTATION (same 3 time points)")
print(f"{'='*70}")

for i in range(min(3, len(train_imputed_sequences[sample_idx]))):
    time_point = train_imputed_sequences[sample_idx][i]
    print(f"\nTime point {i+1} (Hour {time_point.get('Hours', 'N/A')}):")
    
    non_nan = sum(1 for k, v in time_point.items() 
                  if k != 'Hours' and pd.notna(v))
    print(f"  Non-missing: {non_nan}/41")
    
    # Show same parameters
    print(f"  Age: {time_point.get('Age', 'NaN'):.2f}")
    print(f"  HR: {time_point.get('HR', 'NaN'):.2f}")
    print(f"  Temp: {time_point.get('Temp', 'NaN'):.2f}")
    print(f"  Glucose: {time_point.get('Glucose', 'NaN'):.2f}")

# Save imputed sequences
print(f"\n{'='*70}")
print("SAVING IMPUTED SEQUENCES")
print(f"{'='*70}")

with open('p2c_train_imputed.pkl', 'wb') as f:
    pickle.dump({
        'sequences': train_imputed_sequences,
        'patient_ids': train_data['patient_ids'],
        'sequence_lengths': train_data['sequence_lengths'],
        'global_medians': global_medians
    }, f)
print("✓ Saved: p2c_train_imputed.pkl")

with open('p2c_test_imputed.pkl', 'wb') as f:
    pickle.dump({
        'sequences': test_imputed_sequences,
        'patient_ids': test_data['patient_ids'],
        'sequence_lengths': test_data['sequence_lengths'],
        'global_medians': global_medians
    }, f)
print("✓ Saved: p2c_test_imputed.pkl")

print(f"\n{'='*70}")
print("P2(c) COMPLETE!")
print(f"{'='*70}")
print("\nImputation strategy summary:")
print("  ✓ Forward-fill: Carry last observed value forward in time")
print("  ✓ Global median: For never-observed values")
print("  ✓ All 41 parameters now have values at each time point")
print(f"\nFiles saved:")
print(f"  - p2c_train_imputed.pkl ({len(train_imputed_sequences)} patients)")
print(f"  - p2c_test_imputed.pkl ({len(test_imputed_sequences)} patients)")