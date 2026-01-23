# Classical Chinese Instruction Tuning with QLoRA

Fine-tuning Qwen3-4B for Classical Chinese translation using QLoRA (Quantized Low-Rank Adaptation).

## Environment Setup

### Prerequisites
- Python 3.10 or higher
- CUDA-capable GPU with 8GB+ VRAM (16GB recommended)
- 20GB+ disk space
- Linux or macOS (Windows with WSL2)

### Step 1: Install Dependencies

Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required packages:
```bash
pip install --upgrade pip
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.51.0
pip install bitsandbytes==0.44.1
pip install peft==0.13.0
pip install accelerate
pip install datasets==3.0.1
pip install scipy
pip install scikit-learn
pip install tqdm
pip install matplotlib
pip install seaborn
pip install gdown
```

**Note**: Adjust the PyTorch installation command based on your CUDA version. Check [PyTorch website](https://pytorch.org/get-started/locally/) for the appropriate command.

### Step 2: Verify Installation

Test if your GPU is accessible:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

Expected output should show `CUDA available: True` and your GPU name.

## Project Structure

```
r13921031/
├── README.md                      # This file
├── report.pdf                     # Project report
├── download.sh                    # Download trained adapter
├── run.sh                         # Inference wrapper script
├── utils.py                       # Utility functions
├── evaluate_and_plot.py           # Evaluation and plotting script
├── code/                          # All training and analysis code
│   ├── train.py                   # Training script
│   ├── predict.py                 # Inference script
│   ├── training_kaggle.ipynb      # Kaggle notebook version
│   └── zero_shot_few_shot.py      # Zero-shot and few-shot analysis
└── adapter_checkpoint/            # Downloaded/trained adapter
    ├── adapter_config.json
    └── adapter_model.safetensors
```

## Training

### Step 1: Prepare Training Data

Place your training data in the project directory:
```
r13921031/
├── train.json              # 10,000 training samples
├── public_test.json        # 250 test samples
└── ...
```

The JSON format should be:
```json
[
  {
    "instruction": "將下文翻譯成現代文：...",
    "output": "..."
  },
  ...
]
```

### Step 2: Run Training

Navigate to the code directory and run training:

```bash
cd code
python train.py --train_file ../train.json --output_dir ../qlora_adapter
```

**Command-line Arguments:**
- `--train_file`: Path to training JSON file (default: `./train.json`)
- `--output_dir`: Directory to save adapter (default: `./qlora_adapter`)
- `--num_epochs`: Number of training epochs (default: `3`)
- `--batch_size`: Batch size per device (default: `4`)
- `--learning_rate`: Learning rate (default: `2e-4`)

**Full example with custom parameters:**
```bash
cd code
python train.py \
    --train_file ../train.json \
    --output_dir ../qlora_adapter \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4
```

### Step 3: Alternative - Using Kaggle Notebook

You can also train using the provided Kaggle notebook:

1. Upload `code/training_kaggle.ipynb` to Kaggle
2. Add your training data as a dataset
3. Enable GPU accelerator (P100 or T4)
4. Run all cells
5. Download the generated `adapter_checkpoint.zip`

### Step 4: Monitor Training

Training progress will be displayed in the terminal:
```
[1/6] Loading model and tokenizer...
✓ Model and tokenizer loaded

[2/6] Configuring LoRA...
trainable params: 26,091,520 || all params: 3,745,902,592 || trainable%: 0.6967
✓ LoRA configured

[3/6] Preparing training data...
✓ Dataset prepared: 10000 samples

[4/6] Configuring training arguments...
✓ Training arguments configured
  Steps per epoch: 625
  Total steps: 1875
  Estimated time: ~219 minutes

[5/6] Initializing trainer...
✓ Trainer initialized

[6/6] Starting training...
Training: 100%|████████████| 1875/1875 [3:39:00<00:00, 7.02s/it]

✓ TRAINING COMPLETED SUCCESSFULLY!
Final training loss: 0.8234
```

**Training Time Estimates:**
- On P100 (16GB): ~3-4 hours for 3 epochs
- On T4 (16GB): ~4-5 hours for 3 epochs
- On RTX 3090 (24GB): ~2-3 hours for 3 epochs

### Step 5: Prepare Checkpoint for Submission

After training, the adapter will be saved to `./qlora_adapter/`. Move it to the submission location:

```bash
cd ..
mv qlora_adapter adapter_checkpoint
```

Or if you used a custom output directory, rename it:
```bash
mv your_output_dir adapter_checkpoint
```

A ZIP archive `adapter_checkpoint.zip` is automatically created during training. Upload this to Google Drive:

1. Upload `adapter_checkpoint.zip` to Google Drive
2. Right-click → Get link → Change to "Anyone with the link"
3. Copy the file ID from the sharing link:
   ```
   https://drive.google.com/file/d/1abc...xyz/view?usp=sharing
                                    ^^^ This is the file ID
   ```
4. Update `download.sh` with the file ID:
   ```bash
   FILE_ID="1abc...xyz"
   ```

## Inference

### Step 1: Download Trained Adapter

If you need to download a pre-trained adapter:
```bash
bash download.sh
```

This will download and extract the adapter checkpoint to `./adapter_checkpoint/`.

### Step 2: Run Inference

From the project root directory:

```bash
bash run.sh \
    <base_model_path> \
    <adapter_path> \
    <input_json> \
    <output_json>
```

**Example with Hugging Face model:**
```bash
bash run.sh \
    "Qwen/Qwen3-4B" \
    ./adapter_checkpoint \
    ./public_test.json \
    ./predictions.json
```

**Example with local model:**
```bash
bash run.sh \
    /path/to/downloaded/Qwen3-4B \
    ./adapter_checkpoint \
    ./public_test.json \
    ./predictions.json
```

### Step 3: View Results

The predictions will be saved to `predictions.json`:
```json
[
  {
    "id": "0",
    "output": "翻譯結果..."
  },
  ...
]
```

## Evaluation and Analysis

### Calculate Perplexity and Generate Plots

Run the evaluation and plotting script:
```bash
python evaluate_and_plot.py \
    --base_model_path Qwen/Qwen3-4B \
    --adapter_path ./adapter_checkpoint \
    --test_data_path ./public_test.json
```

This will:
- Calculate model perplexity
- Generate training loss curves
- Create performance comparison plots
- Save figures for the report

### Zero-shot and Few-shot Analysis

To reproduce the zero-shot and few-shot experiments from the report:

```bash
cd code
python zero_shot_few_shot.py \
    --base_model_path Qwen/Qwen3-4B \
    --test_data_path ../public_test.json
```

This script analyzes:
- Zero-shot performance of the base model
- Few-shot learning with different numbers of examples (1-shot, 3-shot, 5-shot)
- Comparison with fine-tuned model performance

## Training Details

### Model Architecture
- **Base Model**: Qwen3-4B (4 billion parameters)
- **Fine-tuning Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Quantization**: 4-bit NF4 with double quantization
- **Compute Dtype**: bfloat16

### LoRA Configuration
- **Rank (r)**: 16
- **Alpha**: 32
- **Dropout**: 0.05
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Trainable Parameters**: ~26M (0.7% of total parameters)

### Training Hyperparameters
- **Batch Size**: 4 per device
- **Gradient Accumulation Steps**: 4
- **Effective Batch Size**: 16
- **Learning Rate**: 2e-4
- **Optimizer**: Paged AdamW 8-bit
- **LR Scheduler**: Cosine with warmup
- **Warmup Steps**: 100
- **Training Epochs**: 3
- **Max Sequence Length**: 512
- **Mixed Precision**: bfloat16

### Hardware Requirements
- **Minimum**: NVIDIA GPU with 8GB VRAM (e.g., RTX 3070, RTX 2080 Ti)
- **Recommended**: NVIDIA GPU with 16GB+ VRAM (e.g., P100, V100, RTX 3090, RTX 4090)
- **Training Memory Usage**: ~14GB VRAM
- **Inference Memory Usage**: ~6GB VRAM

## Code Organization

### `code/train.py`
Main training script with:
- Model loading and quantization
- LoRA configuration
- Dataset preparation and tokenization
- Training loop with Hugging Face Trainer
- Checkpoint saving

### `code/predict.py`
Inference script that:
- Loads base model and LoRA adapter
- Processes input JSON files
- Generates translations
- Saves predictions in required format

### `code/training_kaggle.ipynb`
Jupyter notebook version for Kaggle environment:
- Cell-by-cell execution
- Kaggle-specific paths and configurations
- Interactive training monitoring

### `code/zero_shot_few_shot.py`
Analysis script for:
- Zero-shot baseline evaluation
- Few-shot prompting experiments (1-shot to 5-shot)
- Performance comparison
- Statistical analysis

### `utils.py`
Shared utility functions:
- `get_prompt()`: Format instructions as prompts
- `get_bnb_config()`: Configure 4-bit quantization
- Data loading and preprocessing helpers

### `evaluate_and_plot.py`
Evaluation and visualization:
- Perplexity calculation
- Training curve plotting
- Performance metric visualization
- Figure generation for report

### `run.sh`
Wrapper script that:
- Validates input arguments
- Calls `predict.py` with proper parameters
- Handles error checking

### `download.sh`
Download script that:
- Downloads adapter from Google Drive
- Extracts checkpoint files
- Verifies file integrity

## Troubleshooting

### Out of Memory (OOM) Errors

If you encounter OOM during training, modify `code/train.py`:

1. **Reduce batch size**:
   ```python
   BATCH_SIZE = 2  # Reduced from 4
   ```

2. **Increase gradient accumulation**:
   ```python
   GRADIENT_ACCUMULATION_STEPS = 8  # Increased from 4
   ```

3. **Reduce sequence length**:
   ```python
   MAX_LENGTH = 384  # Reduced from 512
   ```

4. **Use smaller LoRA rank**:
   ```python
   LORA_R = 8  # Reduced from 16
   ```

### Import Errors

If you get module import errors:
```bash
pip install --upgrade transformers peft bitsandbytes accelerate
```

### CUDA Version Mismatch

If you see CUDA-related errors:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Replace `cu121` with your CUDA version (e.g., `cu118` for CUDA 11.8).

### Download Issues

If `download.sh` fails:
1. Check internet connectivity
2. Verify the Google Drive link is publicly accessible
3. Ensure the file ID in `download.sh` is correct
4. Try manual download from Google Drive

### Permission Errors

If you get permission denied errors:
```bash
chmod +x run.sh download.sh
```

## Reproducibility Notes

To reproduce the exact results:

1. **Use the same random seed**: The training script uses the default random seed from Transformers (42).

2. **Same hardware**: Results may vary slightly on different GPUs due to floating-point precision differences.

3. **Same software versions**: Use the exact versions specified in the installation section.

4. **Same data**: Use the provided `train.json` with 10,000 samples.

5. **Same hyperparameters**: Use the default hyperparameters in `code/train.py`.

6. **Training order**: Train for exactly 3 epochs as specified.

## Step-by-Step Reproduction Guide for TAs

To reproduce the complete results from scratch:

```bash
# 1. Clone/extract the submission
cd r13921031

# 2. Install dependencies
pip install torch==2.4.1 transformers==4.51.0 bitsandbytes==0.44.1 \
    peft==0.13.0 accelerate datasets==3.0.1 scipy scikit-learn \
    tqdm matplotlib seaborn gdown

# 3. Verify GPU
python -c "import torch; print(torch.cuda.is_available())"

# 4. Download training data (if not provided)
# Place train.json and public_test.json in r13921031/

# 5. Train model
cd code
python train.py --train_file ../train.json --output_dir ../qlora_adapter
cd ..

# 6. Move trained adapter to submission location
mv qlora_adapter adapter_checkpoint

# 7. Run inference
bash run.sh "Qwen/Qwen3-4B" ./adapter_checkpoint ./public_test.json ./predictions.json

# 8. Evaluate and plot
python evaluate_and_plot.py \
    --base_model_path "Qwen/Qwen3-4B" \
    --adapter_path ./adapter_checkpoint \
    --test_data_path ./public_test.json

# 9. Run zero-shot/few-shot analysis
cd code
python zero_shot_few_shot.py \
    --base_model_path "Qwen/Qwen3-4B" \
    --test_data_path ../public_test.json
```

Expected training time: ~3-4 hours on P100 GPU

## Submission Checklist

- [x] `README.md` with complete setup and training instructions
- [x] `report.pdf` - Project report
- [x] `download.sh` - Downloads adapter checkpoint (< 1 hour)
- [x] `run.sh` - Inference wrapper
- [x] `utils.py` - Utility functions
- [x] `evaluate_and_plot.py` - Evaluation script
- [x] `code/train.py` - Training script
- [x] `code/predict.py` - Inference script
- [x] `code/training_kaggle.ipynb` - Kaggle notebook
- [x] `code/zero_shot_few_shot.py` - Analysis script
- [x] `adapter_checkpoint/` - Trained adapter weights
  - [x] `adapter_config.json`
  - [x] `adapter_model.safetensors`

## Important Notes

1. **Do NOT include** the base Qwen3-4B model in your submission
2. **Only submit** the LoRA adapter weights (< 100MB)
3. Ensure `download.sh` completes within **1 hour**
4. Keep Google Drive links **valid for 3+ weeks** after deadline
5. All code should be in the `code/` directory
6. Test your `download.sh` and `run.sh` on a fresh environment before submission

## Contact

For questions about the assignment, please refer to:
- Course materials on NTU COOL
- Office hours with TAs
- Course discussion forum

---

**Course**: Machine Learning (2024 Fall)  
**Institution**: Department of Computer Science and Information Engineering, National Taiwan University  
**Student ID**: r13921031