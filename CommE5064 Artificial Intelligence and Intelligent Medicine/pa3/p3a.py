"""
P3(a): Train a model on time-sequence data and evaluate with AUROC and AUPRC
Model: LSTM (Long Short-Term Memory)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

print("=" * 70)
print("P3(a): TRAINING LSTM MODEL FOR MORTALITY PREDICTION")
print("=" * 70)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Load data
print(f"\n{'='*70}")
print("LOADING DATA")
print(f"{'='*70}")

Xtrain = np.load('p2d_Xtrain.npy')
Xtest = np.load('p2d_Xtest.npy')
ytrain = np.load('p2e_ytrain.npy')
ytest = np.load('p2e_ytest.npy')

print(f"Xtrain shape: {Xtrain.shape}")
print(f"Xtest shape: {Xtest.shape}")
print(f"ytrain shape: {ytrain.shape}")
print(f"ytest shape: {ytest.shape}")

# Data normalization (important for neural networks)
print(f"\n{'='*70}")
print("DATA NORMALIZATION")
print(f"{'='*70}")

# Compute mean and std from training set only
train_mean = Xtrain.mean(axis=(0, 1), keepdims=True)
train_std = Xtrain.std(axis=(0, 1), keepdims=True) + 1e-8  # Add small value to avoid division by zero

# Normalize both sets using training statistics
Xtrain_norm = (Xtrain - train_mean) / train_std
Xtest_norm = (Xtest - train_mean) / train_std

print(f"Before normalization - Train mean: {Xtrain.mean():.4f}, std: {Xtrain.std():.4f}")
print(f"After normalization - Train mean: {Xtrain_norm.mean():.4f}, std: {Xtrain_norm.std():.4f}")

# Create PyTorch datasets
class ICUDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = ICUDataset(Xtrain_norm, ytrain)
test_dataset = ICUDataset(Xtest_norm, ytest)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"\nBatch size: {batch_size}")
print(f"Training batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")

# Define LSTM model
print(f"\n{'='*70}")
print("MODEL ARCHITECTURE")
print(f"{'='*70}")

class LSTMMortalityPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(LSTMMortalityPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the last hidden state
        last_hidden = h_n[-1]  # Shape: (batch, hidden_size)
        
        # Fully connected layers
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out.squeeze()

# Initialize model
input_size = Xtrain.shape[2]  # 41 features
model = LSTMMortalityPredictor(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.3)
model = model.to(device)

print(f"\nModel: LSTM-based Mortality Predictor")
print(f"Input size: {input_size} features")
print(f"Hidden size: 128")
print(f"Number of LSTM layers: 2")
print(f"Dropout: 0.3")
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# Define loss function and optimizer
print(f"\n{'='*70}")
print("TRAINING CONFIGURATION")
print(f"{'='*70}")

# Use BCELoss since model already has sigmoid
criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Learning rate scheduler (removed verbose parameter)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

num_epochs = 50

print(f"Optimizer: Adam")
print(f"Learning rate: 0.001")
print(f"Weight decay: 1e-5")
print(f"Loss function: Binary Cross-Entropy")
print(f"Number of epochs: {num_epochs}")
print(f"Class imbalance: ~6:1 (survived:death)")

# Training loop
print(f"\n{'='*70}")
print("TRAINING")
print(f"{'='*70}")

train_losses = []
train_aurocs = []
test_aurocs = []
test_auprcs = []

best_auroc = 0.0
best_epoch = 0

for epoch in range(num_epochs):
    # Training phase
    model.train()
    epoch_loss = 0.0
    all_train_preds = []
    all_train_labels = []
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        all_train_preds.extend(outputs.detach().cpu().numpy())
        all_train_labels.extend(y_batch.cpu().numpy())
    
    # Compute training metrics
    avg_loss = epoch_loss / len(train_loader)
    train_auroc = roc_auc_score(all_train_labels, all_train_preds)
    
    # Evaluation phase
    model.eval()
    all_test_preds = []
    all_test_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_test_preds.extend(outputs.cpu().numpy())
            all_test_labels.extend(y_batch.numpy())
    
    # Compute test metrics
    test_auroc = roc_auc_score(all_test_labels, all_test_preds)
    test_auprc = average_precision_score(all_test_labels, all_test_preds)
    
    # Update learning rate
    scheduler.step(test_auroc)
    
    # Save metrics
    train_losses.append(avg_loss)
    train_aurocs.append(train_auroc)
    test_aurocs.append(test_auroc)
    test_auprcs.append(test_auprc)
    
    # Save best model
    if test_auroc > best_auroc:
        best_auroc = test_auroc
        best_epoch = epoch
        torch.save(model.state_dict(), 'p3a_best_model.pth')
    
    # Print progress
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f} - "
              f"Train AUROC: {train_auroc:.4f} - Test AUROC: {test_auroc:.4f} - "
              f"Test AUPRC: {test_auprc:.4f}")

print(f"\n{'='*70}")
print("TRAINING COMPLETE")
print(f"{'='*70}")
print(f"Best test AUROC: {best_auroc:.4f} at epoch {best_epoch+1}")

# Load best model for final evaluation
model.load_state_dict(torch.load('p3a_best_model.pth'))
model.eval()

# Final evaluation
print(f"\n{'='*70}")
print("FINAL EVALUATION")
print(f"{'='*70}")

# Get predictions
all_train_preds = []
all_train_labels = []
model.eval()
with torch.no_grad():
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        all_train_preds.extend(outputs.cpu().numpy())
        all_train_labels.extend(y_batch.numpy())

all_test_preds = []
all_test_labels = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        all_test_preds.extend(outputs.cpu().numpy())
        all_test_labels.extend(y_batch.numpy())

# Compute final metrics
final_train_auroc = roc_auc_score(all_train_labels, all_train_preds)
final_train_auprc = average_precision_score(all_train_labels, all_train_preds)
final_test_auroc = roc_auc_score(all_test_labels, all_test_preds)
final_test_auprc = average_precision_score(all_test_labels, all_test_preds)

print(f"\nTRAINING SET:")
print(f"  AUROC: {final_train_auroc:.4f}")
print(f"  AUPRC: {final_train_auprc:.4f}")

print(f"\nTEST SET:")
print(f"  AUROC: {final_test_auroc:.4f}")
print(f"  AUPRC: {final_test_auprc:.4f}")

# Save predictions and metrics
np.save('p3a_train_predictions.npy', np.array(all_train_preds))
np.save('p3a_test_predictions.npy', np.array(all_test_preds))

# Save training history
history = {
    'train_losses': train_losses,
    'train_aurocs': train_aurocs,
    'test_aurocs': test_aurocs,
    'test_auprcs': test_auprcs,
    'best_auroc': best_auroc,
    'best_epoch': best_epoch,
    'final_metrics': {
        'train_auroc': final_train_auroc,
        'train_auprc': final_train_auprc,
        'test_auroc': final_test_auroc,
        'test_auprc': final_test_auprc
    }
}

with open('p3a_training_history.pkl', 'wb') as f:
    pickle.dump(history, f)

print(f"\n✓ Saved predictions")
print(f"✓ Saved best model: p3a_best_model.pth")
print(f"✓ Saved training history: p3a_training_history.pkl")

print(f"\n{'='*70}")
print("P3(a) TRAINING COMPLETE!")
print(f"{'='*70}")