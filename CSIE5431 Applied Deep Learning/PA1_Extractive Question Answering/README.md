# Chinese Extractive Question Answering

This repository contains the implementation for a two-stage Chinese extractive QA system using BERT-based models.

## Model Architecture

The system uses a two-stage approach:
1. **Paragraph Selection**: Multiple choice model to select the relevant paragraph from 4 candidates
2. **Span Selection**: Extractive QA model to identify the answer span within the selected paragraph

## Models Used

- **Paragraph Selection Model**: `bert-base-chinese`
  - Max sequence length: 512
  - Training epochs: 1
  - Learning rate: 3e-5
  - Batch size: 2 (effective)

- **Span Selection Model**: `hfl/chinese-bert-wwm-ext`
  - Max sequence length: 384
  - Training epochs: 5 (best performing checkpoint)
  - Learning rate: 2e-5
  - Batch size: 2 (effective)
  - Weight decay: 0.01

## Directory Structure

```
.
├── README.md
├── download.sh
├── run.sh
├── inference.py
├── train_paragraph_selection.py
├── train_span_selection.py
├── preprocess_data.py
└── requirements.txt
```

## Environment Setup

### Requirements
```
Python 3.10
torch >= 2.0.0
transformers == 4.50.0
datasets == 2.21.0
accelerate == 0.34.2
pandas
numpy
tqdm
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Training Instructions

### Step 1: Data Preprocessing

First, prepare the data in the correct format:

```bash
python preprocess_data.py \
    --context_file context.json \
    --train_file train.json \
    --valid_file valid.json \
    --test_file test.json \
    --output_dir processed_data
```

This will create:
- `processed_data/para_train.json` - Paragraph selection training data
- `processed_data/para_valid.json` - Paragraph selection validation data
- `processed_data/qa_train.json` - Span selection training data
- `processed_data/qa_valid.json` - Span selection validation data

### Step 2: Train Paragraph Selection Model

```bash
python train_paragraph_selection.py \
    --train_file processed_data/para_train.json \
    --validation_file processed_data/para_valid.json \
    --model_name_or_path bert-base-chinese \
    --max_seq_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --output_dir paragraph_model \
    --seed 42
```

**Training time**: ~1.5 hours on RTX 2080 Ti

### Step 3: Train Span Selection Model

```bash
python train_span_selection.py \
    --train_file processed_data/qa_train.json \
    --validation_file processed_data/qa_valid.json \
    --model_name_or_path hfl/chinese-bert-wwm-ext \
    --max_seq_length 384 \
    --doc_stride 128 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --num_train_epochs 4 \
    --output_dir span_model_epoch_4 \
    --seed 42
```

**Training time**: ~6 hours on RTX 2080 Ti

**Note**: The training script saves checkpoints after each epoch. We found epoch 4 performed best on validation.

### Training on Combined Data

For better performance, we combined train and valid data for the final model training:

```python
# In train_span_selection.py, modify data loading:
combined_train_data = train_data + valid_data
```

This increased training examples from 21,714 to 24,723.

## Inference

### Using the Provided Scripts

```bash
# Download models
bash download.sh

# Run inference
bash run.sh /path/to/context.json /path/to/test.json /path/to/prediction.csv
```

### Manual Inference

```bash
python inference.py \
    --context_file context.json \
    --test_file test.json \
    --paragraph_model ./paragraph_model \
    --span_model ./span_model_epoch_4 \
    --output_file prediction.csv \
    --batch_size 16
```

## Performance

- **Public Leaderboard**: 0.76+ EM
- **Private Leaderboard**: 0.76+ EM
- **Strong Baseline**: 0.74853 EM

## Key Design Decisions

1. **Model Selection**: Used `hfl/chinese-bert-wwm-ext` for span selection as it has better Chinese language understanding through whole word masking

2. **Sequence Length**: Used 384 for span model (vs 512) to balance performance and memory usage

3. **Training Strategy**: Combined train and validation data for final training to maximize training examples

4. **Checkpoint Selection**: Tested multiple epoch checkpoints and found epoch 4 performed best, avoiding both underfitting and overfitting

5. **Post-processing**: Used n_best_size=20 and max_answer_length=30 for answer extraction

## Troubleshooting

### Out of Memory
Reduce batch size to 1 and increase gradient accumulation steps:
```bash
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 4
```

### Slow Training
Ensure GPU is being used:
```python
import torch
print(torch.cuda.is_available())  # Should print True
```

### Poor Performance
- Check that models are loaded correctly
- Verify sequence lengths match training configuration
- Ensure preprocessing steps are consistent between training and inference

## References

- HuggingFace Transformers: https://huggingface.co/docs/transformers
- Chinese BERT WWM: https://github.com/ymcui/Chinese-BERT-wwm
- Training scripts adapted from HuggingFace examples

## Contact

For questions or issues, please refer to the course materials or contact the teaching assistants.
