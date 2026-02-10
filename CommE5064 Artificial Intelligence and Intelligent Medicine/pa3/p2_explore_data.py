"""
P2: Exploration of PhysioNet ICU Dataset
Understanding the data structure before processing
"""

import os
import pandas as pd
import numpy as np

print("=" * 70)
print("P2: ICU DATASET EXPLORATION")
print("=" * 70)

# Check directory structure
print("\n1. DIRECTORY STRUCTURE:")
print("-" * 70)

set_a_path = 'set-a'
set_b_path = 'set-b'

if os.path.exists(set_a_path):
    set_a_files = sorted([f for f in os.listdir(set_a_path) if f.endswith('.txt')])
    print(f"Set A (Training): {len(set_a_files)} patient files")
    print(f"First 5 files: {set_a_files[:5]}")
    print(f"Last 5 files: {set_a_files[-5:]}")
else:
    print(f"ERROR: {set_a_path} not found")

print()

if os.path.exists(set_b_path):
    set_b_files = sorted([f for f in os.listdir(set_b_path) if f.endswith('.txt')])
    print(f"Set B (Test): {len(set_b_files)} patient files")
    print(f"First 5 files: {set_b_files[:5]}")
    print(f"Last 5 files: {set_b_files[-5:]}")
else:
    print(f"ERROR: {set_b_path} not found")

# Examine a sample patient file
print("\n2. SAMPLE PATIENT FILE STRUCTURE:")
print("-" * 70)

sample_file = os.path.join(set_a_path, set_a_files[0]) if os.path.exists(set_a_path) else None

if sample_file and os.path.exists(sample_file):
    print(f"Examining: {set_a_files[0]}")
    print()
    
    # Read first 30 lines
    with open(sample_file, 'r') as f:
        lines = [f.readline().strip() for _ in range(30)]
    
    print("First 30 lines:")
    for i, line in enumerate(lines, 1):
        print(f"{i:3d}: {line}")
    
    # Load full file
    print("\n" + "-" * 70)
    df = pd.read_csv(sample_file)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 10 rows:")
    print(df.head(10))
    
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nUnique parameters recorded:")
    if 'Parameter' in df.columns:
        params = df['Parameter'].unique()
        print(f"Total unique parameters: {len(params)}")
        print(f"Parameters: {sorted(params)}")
    
    print(f"\nTime range:")
    if 'Time' in df.columns:
        print(f"Min time: {df['Time'].min()}")
        print(f"Max time: {df['Time'].max()}")
        print(f"Unique time points: {df['Time'].nunique()}")

# Check outcomes files
print("\n3. OUTCOMES FILES:")
print("-" * 70)

outcomes_a = 'Outcomes-a.txt'
outcomes_b = 'Outcomes-b.txt'

for outcomes_file in [outcomes_a, outcomes_b]:
    if os.path.exists(outcomes_file):
        print(f"\n{outcomes_file}:")
        df_outcomes = pd.read_csv(outcomes_file)
        print(f"Shape: {df_outcomes.shape}")
        print(f"Columns: {list(df_outcomes.columns)}")
        print(f"\nFirst 10 rows:")
        print(df_outcomes.head(10))
        
        # Check for outcome distribution
        if 'In-hospital_death' in df_outcomes.columns:
            print(f"\nOutcome distribution:")
            print(df_outcomes['In-hospital_death'].value_counts())
            print(f"Death rate: {df_outcomes['In-hospital_death'].mean():.2%}")
    else:
        print(f"\n{outcomes_file}: NOT FOUND")

print("\n" + "=" * 70)
print("EXPLORATION COMPLETE")
print("=" * 70)