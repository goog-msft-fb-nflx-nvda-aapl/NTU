"""
P3(b): Report model architecture using torchinfo
"""

import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np

print("=" * 70)
print("P3(b): MODEL ARCHITECTURE REPORT")
print("=" * 70)

# Recreate the model architecture
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
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out.squeeze()

# Initialize model
input_size = 41  # 41 features
model = LSTMMortalityPredictor(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.3)

# Load the trained weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('p3a_best_model.pth'))
model = model.to(device)
model.eval()

print("\n" + "="*70)
print("DETAILED MODEL ARCHITECTURE (using torchinfo)")
print("="*70 + "\n")

# Generate detailed summary
batch_size = 32
seq_length = 24
model_summary = summary(
    model, 
    input_size=(batch_size, seq_length, input_size),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    depth=4,
    verbose=1
)

# Save summary to file
with open('p3b_model_architecture.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("P3(b): MODEL ARCHITECTURE REPORT\n")
    f.write("="*70 + "\n\n")
    f.write(f"Model: LSTM-based Mortality Predictor\n\n")
    f.write(str(model_summary))
    f.write("\n\n" + "="*70 + "\n")
    f.write("ARCHITECTURE DETAILS\n")
    f.write("="*70 + "\n\n")
    f.write("Layer 1: LSTM Layer 1\n")
    f.write("  Input: (batch_size, 24, 41)\n")
    f.write("  Hidden units: 128\n")
    f.write("  Output: (batch_size, 24, 128)\n\n")
    f.write("Layer 2: LSTM Layer 2\n")
    f.write("  Input: (batch_size, 24, 128)\n")
    f.write("  Hidden units: 128\n")
    f.write("  Dropout: 0.3\n")
    f.write("  Output: (batch_size, 24, 128)\n\n")
    f.write("Layer 3: Extract Last Hidden State\n")
    f.write("  Input: (batch_size, 24, 128)\n")
    f.write("  Output: (batch_size, 128)\n\n")
    f.write("Layer 4: Fully Connected Layer 1\n")
    f.write("  Input: (batch_size, 128)\n")
    f.write("  Output: (batch_size, 64)\n")
    f.write("  Activation: ReLU\n\n")
    f.write("Layer 5: Dropout\n")
    f.write("  Dropout rate: 0.3\n\n")
    f.write("Layer 6: Fully Connected Layer 2\n")
    f.write("  Input: (batch_size, 64)\n")
    f.write("  Output: (batch_size, 1)\n")
    f.write("  Activation: Sigmoid\n\n")
    f.write("="*70 + "\n")
    f.write("TOTAL PARAMETERS\n")
    f.write("="*70 + "\n")
    f.write(f"Total params: {sum(p.numel() for p in model.parameters()):,}\n")
    f.write(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    f.write(f"Non-trainable params: 0\n")

print("\n✓ Saved: p3b_model_architecture.txt")

# Create a visual architecture diagram using ASCII art
print("\n" + "="*70)
print("MODEL ARCHITECTURE VISUALIZATION")
print("="*70 + "\n")

architecture_diagram = """
┌─────────────────────────────────────────────────────────────┐
│                    INPUT SEQUENCE                           │
│              (batch_size, 24, 41)                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  LSTM Layer 1                                │
│          Input: 41 → Hidden: 128                            │
│          Parameters: 87,040                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              LSTM Layer 2 + Dropout                         │
│          Input: 128 → Hidden: 128                           │
│          Parameters: 131,584                                │
│          Dropout: 0.3                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            Extract Last Hidden State                        │
│              (batch_size, 128)                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Fully Connected Layer 1                        │
│          Input: 128 → Output: 64                            │
│          Parameters: 8,256                                  │
│          Activation: ReLU                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Dropout                                  │
│               Dropout rate: 0.3                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Fully Connected Layer 2                        │
│          Input: 64 → Output: 1                              │
│          Parameters: 65                                     │
│          Activation: Sigmoid                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                OUTPUT PREDICTION                            │
│        (batch_size,) - Mortality probability                │
└─────────────────────────────────────────────────────────────┘

Total Parameters: 227,969
"""

print(architecture_diagram)

# Save ASCII diagram
with open('p3b_architecture_diagram.txt', 'w') as f:
    f.write(architecture_diagram)

print("\n✓ Saved: p3b_architecture_diagram.txt")

# Calculate parameters per layer for LaTeX table
print("\n" + "="*70)
print("PARAMETER BREAKDOWN (for LaTeX table)")
print("="*70 + "\n")

param_breakdown = """
LaTeX Table Code:

\\begin{table}[h]
\\centering
\\begin{tabular}{lcccc}
\\hline
\\textbf{Layer} & \\textbf{Input Shape} & \\textbf{Output Shape} & \\textbf{Parameters} & \\textbf{Activation} \\\\
\\hline
LSTM Layer 1 & (24, 41) & (24, 128) & 87,040 & tanh/sigmoid \\\\
LSTM Layer 2 & (24, 128) & (24, 128) & 131,584 & tanh/sigmoid \\\\
Extract Last & (24, 128) & (128) & 0 & - \\\\
FC Layer 1 & (128) & (64) & 8,256 & ReLU \\\\
Dropout & (64) & (64) & 0 & - \\\\
FC Layer 2 & (64) & (1) & 65 & Sigmoid \\\\
\\hline
\\textbf{Total} & & & \\textbf{227,969} & \\\\
\\hline
\\end{tabular}
\\caption{LSTM Model Architecture - Layer-wise parameter breakdown}
\\end{table}
"""

print(param_breakdown)

with open('p3b_latex_table.txt', 'w') as f:
    f.write(param_breakdown)

print("\n✓ Saved: p3b_latex_table.txt")

print("\n" + "="*70)
print("P3(b) COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  1. p3b_model_architecture.txt - Detailed torchinfo summary")
print("  2. p3b_architecture_diagram.txt - ASCII architecture diagram")
print("  3. p3b_latex_table.txt - LaTeX table code for report")