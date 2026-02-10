"""
P2(b): Create time-sequence vectors for ALL patients
Apply the grouping procedure to all patients in set-a and set-b
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

print("=" * 70)
print("P2(b): CREATING TIME-SEQUENCE VECTORS FOR ALL PATIENTS")
print("=" * 70)

# Define all 41 parameters (42 including Hours)
ALL_PARAMETERS = [
    'RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight',
    'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol',
    'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT',
    'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent', 'Na', 'NIDiasABP',
    'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
    'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
]

def time_to_hours(time_str):
    """Convert time string (HH:MM) to hours as float"""
    try:
        hours, minutes = map(int, time_str.split(':'))
        return hours + minutes / 60.0
    except:
        return np.nan

def parse_patient_file(filepath):
    """Parse patient file and group by time points"""
    df = pd.read_csv(filepath)
    df['Hours'] = df['Time'].apply(time_to_hours)
    
    time_data = {}
    for time_hours in df['Hours'].unique():
        if pd.isna(time_hours):
            continue
        
        time_records = df[df['Hours'] == time_hours]
        time_point_dict = {param: np.nan for param in ALL_PARAMETERS}
        time_point_dict['Hours'] = time_hours
        
        for _, row in time_records.iterrows():
            param = row['Parameter']
            value = row['Value']
            if param in ALL_PARAMETERS:
                time_point_dict[param] = value
        
        time_data[time_hours] = time_point_dict
    
    return time_data

def create_sequence_for_patient(time_data, anchor_hour=36, sequence_length=24):
    """Create sequence from (anchor - sequence_length) to anchor"""
    start_hour = anchor_hour - sequence_length
    valid_times = sorted([t for t in time_data.keys() if start_hour <= t <= anchor_hour])
    
    sequence = []
    for time_hour in valid_times:
        time_point_data = time_data[time_hour].copy()
        sequence.append(time_point_data)
    
    return sequence, valid_times

def process_patient_set(set_path, set_name):
    """Process all patients in a set"""
    print(f"\n{'='*70}")
    print(f"Processing {set_name}")
    print(f"{'='*70}")
    
    # Get all patient files
    patient_files = sorted([f for f in os.listdir(set_path) if f.endswith('.txt')])
    print(f"Total patients: {len(patient_files)}")
    
    all_sequences = []
    patient_ids = []
    sequence_lengths = []
    
    # Process each patient
    for patient_file in tqdm(patient_files, desc=f"Processing {set_name}"):
        patient_id = patient_file.replace('.txt', '')
        filepath = os.path.join(set_path, patient_file)
        
        try:
            # Parse and create sequence
            time_data = parse_patient_file(filepath)
            sequence, valid_times = create_sequence_for_patient(time_data, 
                                                                anchor_hour=36, 
                                                                sequence_length=24)
            
            all_sequences.append(sequence)
            patient_ids.append(patient_id)
            sequence_lengths.append(len(sequence))
            
        except Exception as e:
            print(f"\nError processing {patient_id}: {e}")
            continue
    
    return all_sequences, patient_ids, sequence_lengths

# Process training set (set-a)
train_sequences, train_ids, train_lengths = process_patient_set('set-a', 'Set A (Training)')

print(f"\n{'='*70}")
print("TRAINING SET STATISTICS")
print(f"{'='*70}")
print(f"Total patients processed: {len(train_sequences)}")
print(f"Sequence length statistics:")
print(f"  Min: {min(train_lengths)}")
print(f"  Max: {max(train_lengths)}")
print(f"  Mean: {np.mean(train_lengths):.2f}")
print(f"  Median: {np.median(train_lengths):.0f}")
print(f"  Std: {np.std(train_lengths):.2f}")

# Process test set (set-b)
test_sequences, test_ids, test_lengths = process_patient_set('set-b', 'Set B (Test)')

print(f"\n{'='*70}")
print("TEST SET STATISTICS")
print(f"{'='*70}")
print(f"Total patients processed: {len(test_sequences)}")
print(f"Sequence length statistics:")
print(f"  Min: {min(test_lengths)}")
print(f"  Max: {max(test_lengths)}")
print(f"  Mean: {np.mean(test_lengths):.2f}")
print(f"  Median: {np.median(test_lengths):.0f}")
print(f"  Std: {np.std(test_lengths):.2f}")

# Show example sequence (similar to Workshop 3 format)
print(f"\n{'='*70}")
print("EXAMPLE SEQUENCE (First patient from training set)")
print(f"{'='*70}")
print(f"Patient ID: {train_ids[0]}")
print(f"Sequence length: {len(train_sequences[0])} time points")
print(f"\nFirst 3 time points:")

for i, time_point in enumerate(train_sequences[0][:3]):
    print(f"\n  Time point {i+1} (Hour {time_point['Hours']:.2f}):")
    # Show parameters with values
    params_with_values = {k: v for k, v in time_point.items() 
                         if k != 'Hours' and not pd.isna(v)}
    print(f"    Non-missing parameters: {len(params_with_values)}/{len(ALL_PARAMETERS)}")
    # Show first 5
    for j, (k, v) in enumerate(list(params_with_values.items())[:5]):
        print(f"      {k}: {v}")
    if len(params_with_values) > 5:
        print(f"      ... and {len(params_with_values) - 5} more")

# Save sequences
print(f"\n{'='*70}")
print("SAVING SEQUENCES")
print(f"{'='*70}")

# Save as pickle for easy loading
with open('p2b_train_sequences.pkl', 'wb') as f:
    pickle.dump({
        'sequences': train_sequences,
        'patient_ids': train_ids,
        'sequence_lengths': train_lengths
    }, f)
print("Training sequences saved to: p2b_train_sequences.pkl")

with open('p2b_test_sequences.pkl', 'wb') as f:
    pickle.dump({
        'sequences': test_sequences,
        'patient_ids': test_ids,
        'sequence_lengths': test_lengths
    }, f)
print("Test sequences saved to: p2b_test_sequences.pkl")

# Also save one example as readable format (similar to Workshop 3 Figure 3)
example_output = []
for i, seq in enumerate(train_sequences[:3]):  # First 3 patients
    patient_seq = []
    for time_point in seq:
        # Format as list with key info
        patient_seq.append([
            train_ids[i],
            time_point.get('Hours', np.nan),
            time_point.get('Age', np.nan),
            time_point.get('Gender', np.nan),
            time_point.get('Height', np.nan),
            time_point.get('Weight', np.nan),
            time_point.get('HR', np.nan)
        ])
    example_output.append(patient_seq)

# Save as text (similar to Workshop 3 output)
with open('p2b_example_format.txt', 'w') as f:
    f.write("Example sequences (first 3 patients) - Similar to Workshop 3 Figure 3\n")
    f.write("Format: [PatientID, Hours, Age, Gender, Height, Weight, HR]\n")
    f.write("="*70 + "\n\n")
    for i, seq in enumerate(example_output):
        f.write(f"Patient {i+1} (ID: {train_ids[i]}):\n")
        f.write(str(seq[:5]))  # Show first 5 time points
        f.write("\n\n")

print("Example format saved to: p2b_example_format.txt")

print(f"\n{'='*70}")
print("P2(b) COMPLETE - All sequences created!")
print(f"{'='*70}")
print(f"\nSummary:")
print(f"  Training patients: {len(train_sequences)}")
print(f"  Test patients: {len(test_sequences)}")
print(f"  Parameters per time point: {len(ALL_PARAMETERS)}")
print(f"  Average sequence length: {np.mean(train_lengths + test_lengths):.1f} time points")