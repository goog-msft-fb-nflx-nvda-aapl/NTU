"""
P2(e): Extract prediction labels from Outcomes files and show class distributions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

print("=" * 70)
print("P2(e): EXTRACTING PREDICTION LABELS")
print("=" * 70)

# Load metadata to get patient IDs in correct order
with open('p2d_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

train_patient_ids = metadata['train_patient_ids']
test_patient_ids = metadata['test_patient_ids']

print(f"Number of training patients: {len(train_patient_ids)}")
print(f"Number of test patients: {len(test_patient_ids)}")

print(f"\n{'='*70}")
print("LOADING OUTCOME FILES")
print(f"{'='*70}")

# Load outcomes
outcomes_a = pd.read_csv('Outcomes-a.txt')
outcomes_b = pd.read_csv('Outcomes-b.txt')

print(f"Outcomes-a shape: {outcomes_a.shape}")
print(f"Outcomes-b shape: {outcomes_b.shape}")

print(f"\nOutcomes-a columns: {list(outcomes_a.columns)}")
print(f"\nFirst few rows of Outcomes-a:")
print(outcomes_a.head())

# The outcome file has columns: RecordID, SAPS-I, SOFA, Length_of_stay, Survival, In-hospital_death
# We need 'In-hospital_death' column

print(f"\n{'='*70}")
print("EXTRACTING LABELS")
print(f"{'='*70}")

def extract_labels(patient_ids, outcomes_df):
    """
    Extract 'In-hospital_death' labels for each patient ID
    
    Args:
        patient_ids: List of patient IDs (as strings)
        outcomes_df: DataFrame with RecordID and In-hospital_death columns
    
    Returns:
        numpy array of labels (0 or 1)
    """
    labels = []
    
    # Convert RecordID to string for matching
    outcomes_df = outcomes_df.copy()
    outcomes_df['RecordID'] = outcomes_df['RecordID'].astype(str)
    
    # Create a dictionary for fast lookup
    outcome_dict = dict(zip(outcomes_df['RecordID'], outcomes_df['In-hospital_death']))
    
    for patient_id in patient_ids:
        if patient_id in outcome_dict:
            label = outcome_dict[patient_id]
            labels.append(int(label))
        else:
            print(f"Warning: Patient {patient_id} not found in outcomes")
            labels.append(-1)  # Mark as missing
    
    return np.array(labels)

# Extract labels
ytrain = extract_labels(train_patient_ids, outcomes_a)
ytest = extract_labels(test_patient_ids, outcomes_b)

print(f"ytrain shape: {ytrain.shape}")
print(f"ytest shape: {ytest.shape}")

# Check for missing labels
train_missing = (ytrain == -1).sum()
test_missing = (ytest == -1).sum()

if train_missing > 0:
    print(f"\nWarning: {train_missing} training labels missing")
if test_missing > 0:
    print(f"Warning: {test_missing} test labels missing")

print(f"\n{'='*70}")
print("CLASS DISTRIBUTIONS")
print(f"{'='*70}")

# Training set distribution
train_class_0 = (ytrain == 0).sum()
train_class_1 = (ytrain == 1).sum()
train_total = len(ytrain)

print(f"\nTRAINING SET (Set A):")
print(f"  Class 0 (Survived): {train_class_0} ({train_class_0/train_total*100:.2f}%)")
print(f"  Class 1 (Death): {train_class_1} ({train_class_1/train_total*100:.2f}%)")
print(f"  Total: {train_total}")
print(f"  Class imbalance ratio: {train_class_0/train_class_1:.2f}:1")

# Test set distribution
test_class_0 = (ytest == 0).sum()
test_class_1 = (ytest == 1).sum()
test_total = len(ytest)

print(f"\nTEST SET (Set B):")
print(f"  Class 0 (Survived): {test_class_0} ({test_class_0/test_total*100:.2f}%)")
print(f"  Class 1 (Death): {test_class_1} ({test_class_1/test_total*100:.2f}%)")
print(f"  Total: {test_total}")
print(f"  Class imbalance ratio: {test_class_0/test_class_1:.2f}:1")

print(f"\n{'='*70}")
print("VISUALIZATION")
print(f"{'='*70}")

# Create bar plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Training set
axes[0].bar(['Survived (0)', 'Death (1)'], [train_class_0, train_class_1], 
            color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Training Set (Set A) - Class Distribution', fontsize=13, fontweight='bold')
axes[0].set_ylim([0, max(train_class_0, train_class_1) * 1.1])
axes[0].grid(axis='y', alpha=0.3)

# Add counts on bars
for i, (count, pct) in enumerate([(train_class_0, train_class_0/train_total*100), 
                                   (train_class_1, train_class_1/train_total*100)]):
    axes[0].text(i, count + max(train_class_0, train_class_1)*0.02, 
                f'{count}\n({pct:.2f}%)', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)

# Test set
axes[1].bar(['Survived (0)', 'Death (1)'], [test_class_0, test_class_1], 
            color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Test Set (Set B) - Class Distribution', fontsize=13, fontweight='bold')
axes[1].set_ylim([0, max(test_class_0, test_class_1) * 1.1])
axes[1].grid(axis='y', alpha=0.3)

# Add counts on bars
for i, (count, pct) in enumerate([(test_class_0, test_class_0/test_total*100), 
                                   (test_class_1, test_class_1/test_total*100)]):
    axes[1].text(i, count + max(test_class_0, test_class_1)*0.02, 
                f'{count}\n({pct:.2f}%)', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('p2e_class_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: p2e_class_distributions.png")
plt.close()

# Save labels
print(f"\n{'='*70}")
print("SAVING LABELS")
print(f"{'='*70}")

np.save('p2e_ytrain.npy', ytrain)
print(f"✓ Saved: p2e_ytrain.npy")

np.save('p2e_ytest.npy', ytest)
print(f"✓ Saved: p2e_ytest.npy")

# Save summary statistics
summary = {
    'train': {
        'class_0': int(train_class_0),
        'class_1': int(train_class_1),
        'total': int(train_total),
        'class_0_pct': float(train_class_0/train_total*100),
        'class_1_pct': float(train_class_1/train_total*100),
        'imbalance_ratio': float(train_class_0/train_class_1)
    },
    'test': {
        'class_0': int(test_class_0),
        'class_1': int(test_class_1),
        'total': int(test_total),
        'class_0_pct': float(test_class_0/test_total*100),
        'class_1_pct': float(test_class_1/test_total*100),
        'imbalance_ratio': float(test_class_0/test_class_1)
    }
}

with open('p2e_label_summary.pkl', 'wb') as f:
    pickle.dump(summary, f)
print(f"✓ Saved: p2e_label_summary.pkl")

print(f"\n{'='*70}")
print("P2(e) COMPLETE!")
print(f"{'='*70}")
print("\nSummary:")
print(f"  ✓ ytrain: {ytrain.shape} - {train_class_1} deaths ({train_class_1/train_total*100:.2f}%)")
print(f"  ✓ ytest: {ytest.shape} - {test_class_1} deaths ({test_class_1/test_total*100:.2f}%)")
print(f"  ✓ Class imbalance: ~6:1 (survived:death)")
print(f"  ✓ Visualization saved")

print(f"\n{'='*70}")
print("P2 COMPLETE! Ready for P3 (Model Training)")
print(f"{'='*70}")