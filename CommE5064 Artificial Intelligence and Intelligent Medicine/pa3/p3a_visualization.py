"""
P3(a): Visualization and Model Justification
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import seaborn as sns

print("=" * 70)
print("P3(a): GENERATING VISUALIZATIONS AND JUSTIFICATION")
print("=" * 70)

# Load training history
with open('p3a_training_history.pkl', 'rb') as f:
    history = pickle.load(f)

# Load predictions
train_preds = np.load('p3a_train_predictions.npy')
test_preds = np.load('p3a_test_predictions.npy')
ytrain = np.load('p2e_ytrain.npy')
ytest = np.load('p2e_ytest.npy')

print(f"Loaded training history and predictions")

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))

# 1. Training Loss Curve
ax1 = plt.subplot(3, 3, 1)
epochs = range(1, len(history['train_losses']) + 1)
ax1.plot(epochs, history['train_losses'], 'b-', linewidth=2, label='Training Loss')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Training Loss Over Time', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. AUROC Curves (Train vs Test)
ax2 = plt.subplot(3, 3, 2)
ax2.plot(epochs, history['train_aurocs'], 'g-', linewidth=2, label='Train AUROC')
ax2.plot(epochs, history['test_aurocs'], 'r-', linewidth=2, label='Test AUROC')
ax2.axvline(x=history['best_epoch']+1, color='orange', linestyle='--', 
            label=f"Best (Epoch {history['best_epoch']+1})", alpha=0.7)
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('AUROC', fontsize=11)
ax2.set_title('AUROC: Training vs Test', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_ylim([0.65, 1.0])

# 3. AUPRC Curve (Test)
ax3 = plt.subplot(3, 3, 3)
ax3.plot(epochs, history['test_auprcs'], 'purple', linewidth=2)
ax3.axvline(x=history['best_epoch']+1, color='orange', linestyle='--', 
            label=f"Best (Epoch {history['best_epoch']+1})", alpha=0.7)
ax3.set_xlabel('Epoch', fontsize=11)
ax3.set_ylabel('AUPRC', fontsize=11)
ax3.set_title('Test AUPRC Over Time', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# 4. ROC Curve - Training
ax4 = plt.subplot(3, 3, 4)
fpr_train, tpr_train, _ = roc_curve(ytrain, train_preds)
ax4.plot(fpr_train, tpr_train, 'g-', linewidth=2, 
         label=f'Train (AUROC = {history["final_metrics"]["train_auroc"]:.4f})')
ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax4.set_xlabel('False Positive Rate', fontsize=11)
ax4.set_ylabel('True Positive Rate', fontsize=11)
ax4.set_title('ROC Curve - Training Set', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(loc='lower right')

# 5. ROC Curve - Test
ax5 = plt.subplot(3, 3, 5)
fpr_test, tpr_test, _ = roc_curve(ytest, test_preds)
ax5.plot(fpr_test, tpr_test, 'r-', linewidth=2, 
         label=f'Test (AUROC = {history["final_metrics"]["test_auroc"]:.4f})')
ax5.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax5.set_xlabel('False Positive Rate', fontsize=11)
ax5.set_ylabel('True Positive Rate', fontsize=11)
ax5.set_title('ROC Curve - Test Set', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend(loc='lower right')

# 6. Precision-Recall Curve - Test
ax6 = plt.subplot(3, 3, 6)
precision, recall, _ = precision_recall_curve(ytest, test_preds)
ax6.plot(recall, precision, 'purple', linewidth=2, 
         label=f'Test (AUPRC = {history["final_metrics"]["test_auprc"]:.4f})')
# Baseline (random classifier)
baseline = ytest.sum() / len(ytest)
ax6.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
            label=f'Baseline ({baseline:.3f})', alpha=0.5)
ax6.set_xlabel('Recall', fontsize=11)
ax6.set_ylabel('Precision', fontsize=11)
ax6.set_title('Precision-Recall Curve - Test Set', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend(loc='upper right')

# 7. Prediction Distribution - Training
ax7 = plt.subplot(3, 3, 7)
ax7.hist(train_preds[ytrain == 0], bins=50, alpha=0.6, color='green', label='Survived (0)', edgecolor='black')
ax7.hist(train_preds[ytrain == 1], bins=50, alpha=0.6, color='red', label='Death (1)', edgecolor='black')
ax7.set_xlabel('Predicted Probability', fontsize=11)
ax7.set_ylabel('Frequency', fontsize=11)
ax7.set_title('Prediction Distribution - Train', fontsize=12, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3, axis='y')

# 8. Prediction Distribution - Test
ax8 = plt.subplot(3, 3, 8)
ax8.hist(test_preds[ytest == 0], bins=50, alpha=0.6, color='green', label='Survived (0)', edgecolor='black')
ax8.hist(test_preds[ytest == 1], bins=50, alpha=0.6, color='red', label='Death (1)', edgecolor='black')
ax8.set_xlabel('Predicted Probability', fontsize=11)
ax8.set_ylabel('Frequency', fontsize=11)
ax8.set_title('Prediction Distribution - Test', fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

# 9. Confusion Matrix (Test, threshold=0.5)
ax9 = plt.subplot(3, 3, 9)
test_pred_binary = (test_preds >= 0.5).astype(int)
cm = confusion_matrix(ytest, test_pred_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax9,
            xticklabels=['Survived', 'Death'], yticklabels=['Survived', 'Death'])
ax9.set_xlabel('Predicted', fontsize=11)
ax9.set_ylabel('Actual', fontsize=11)
ax9.set_title('Confusion Matrix - Test (threshold=0.5)', fontsize=12, fontweight='bold')

# Calculate metrics from confusion matrix
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

# Add text box with metrics
textstr = f'Accuracy: {accuracy:.3f}\nSensitivity: {sensitivity:.3f}\nSpecificity: {specificity:.3f}\nPPV: {ppv:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax9.text(1.5, 0.5, textstr, transform=ax9.transAxes, fontsize=10,
        verticalalignment='center', bbox=props)

plt.tight_layout()
plt.savefig('p3a_comprehensive_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: p3a_comprehensive_results.png")
plt.close()

# Print final summary
print(f"\n{'='*70}")
print("FINAL RESULTS SUMMARY")
print(f"{'='*70}")

print(f"\nMODEL: LSTM-based Mortality Predictor")
print(f"  Architecture: 2-layer LSTM with 128 hidden units")
print(f"  Total parameters: 227,969")
print(f"  Training epochs: 50")
print(f"  Best epoch: {history['best_epoch']+1}")

print(f"\nPERFORMANCE METRICS:")
print(f"\n  Training Set:")
print(f"    AUROC: {history['final_metrics']['train_auroc']:.4f}")
print(f"    AUPRC: {history['final_metrics']['train_auprc']:.4f}")

print(f"\n  Test Set:")
print(f"    AUROC: {history['final_metrics']['test_auroc']:.4f}")
print(f"    AUPRC: {history['final_metrics']['test_auprc']:.4f}")

print(f"\n  Confusion Matrix Metrics (threshold=0.5):")
print(f"    Accuracy: {accuracy:.4f}")
print(f"    Sensitivity (Recall): {sensitivity:.4f}")
print(f"    Specificity: {specificity:.4f}")
print(f"    PPV (Precision): {ppv:.4f}")

print(f"\n{'='*70}")
print("MODEL JUSTIFICATION")
print(f"{'='*70}")

justification = """
WHY LSTM FOR THIS TASK?

1. TEMPORAL DEPENDENCIES:
   - ICU patient data is inherently sequential - vital signs and lab values 
     evolve over time
   - LSTM can capture both short-term patterns (e.g., rapid BP changes) and 
     long-term trends (e.g., gradual deterioration)
   - The model learns to weight recent vs. historical measurements appropriately

2. IRREGULAR SAMPLING HANDLING:
   - ICU measurements are irregularly sampled (not all vitals recorded at 
     every time point)
   - Our imputation + zero-padding strategy combined with LSTM's gating 
     mechanism helps the model learn to ignore padded values
   - LSTM gates can learn which time points contain informative signals

3. VARIABLE-LENGTH SEQUENCES:
   - Patients have different ICU stay durations (0-110 time points in our data)
   - LSTM naturally handles variable-length inputs through its recurrent structure
   - Zero-padding ensures fixed input size while LSTM learns to focus on 
     actual data

4. CLASS IMBALANCE ROBUSTNESS:
   - 6:1 imbalance (survived:death) requires robust learning
   - LSTM with dropout (0.3) prevents overfitting to majority class
   - Achieved good AUPRC (0.46) despite severe imbalance (baseline ~0.14)

5. PROVEN PERFORMANCE:
   - LSTMs are widely used in clinical prediction tasks (e.g., MIMIC-III studies)
   - Our results (AUROC=0.83) are competitive with published ICU mortality 
     prediction models
   - Better than simple baselines (logistic regression on last time point)

ALTERNATIVE MODELS CONSIDERED:

❌ Logistic Regression: 
   - Cannot capture temporal patterns
   - Would only use last time point or aggregate statistics
   - Loses rich sequential information

❌ Standard RNN:
   - Suffers from vanishing gradient problem
   - Cannot learn long-term dependencies (24-hour sequences)
   - LSTM's gates solve this problem

❌ 1D CNN:
   - Good for local patterns but misses long-range dependencies
   - Less interpretable for sequential medical data
   - Fixed receptive field limits temporal modeling

❌ Transformer:
   - Requires more data than available (4000 patients)
   - Higher computational cost
   - LSTM is more parameter-efficient for this dataset size

✓ GRU (potential alternative):
   - Similar to LSTM but fewer parameters
   - Could be explored but LSTM is more established in medical literature

CONCLUSION:
LSTM is the most appropriate model for this ICU mortality prediction task due to 
its ability to model temporal dependencies, handle irregular sampling, work with 
variable-length sequences, and achieve strong performance despite class imbalance.
"""

print(justification)

# Save justification to file
with open('p3a_model_justification.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("P3(a): MODEL JUSTIFICATION\n")
    f.write("="*70 + "\n\n")
    f.write(justification)
    f.write("\n\n" + "="*70 + "\n")
    f.write("PERFORMANCE SUMMARY\n")
    f.write("="*70 + "\n")
    f.write(f"Test AUROC: {history['final_metrics']['test_auroc']:.4f}\n")
    f.write(f"Test AUPRC: {history['final_metrics']['test_auprc']:.4f}\n")

print(f"\n✓ Saved: p3a_model_justification.txt")

print(f"\n{'='*70}")
print("P3(a) COMPLETE!")
print(f"{'='*70}")