# Span Model Only Training with Epoch Checkpoints
# Enhanced span model training with automatic saving after each epoch

# =============================================================================
# CELL 1: Setup and Imports (Same as before)
# =============================================================================

import json
import os
import math
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

import transformers
from transformers import (
    AutoConfig, AutoTokenizer, 
    AutoModelForQuestionAnswering,
    DataCollatorWithPadding,
    get_scheduler, SchedulerType
)

import datasets
from datasets import Dataset

from accelerate import Accelerator
from accelerate.utils import set_seed

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# =============================================================================
# CELL 2: Load Data (Same preprocessing as original)
# =============================================================================

def load_data():
    """Load dataset files from Kaggle input"""
    data_path = "/kaggle/input"
    dataset_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    print(f"Available datasets: {dataset_folders}")
    
    if dataset_folders:
        data_path = os.path.join(data_path, dataset_folders[0])
    
    with open(os.path.join(data_path, 'context.json'), 'r', encoding='utf-8') as f:
        contexts = json.load(f)
    
    with open(os.path.join(data_path, 'train.json'), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(os.path.join(data_path, 'valid.json'), 'r', encoding='utf-8') as f:
        valid_data = json.load(f)
    
    return contexts, train_data, valid_data

def prepare_qa_data(contexts, data):
    """Convert data to SQuAD-like format for extractive QA"""
    converted_data = []
    
    for item in data:
        question = item['question']
        context = contexts[item['relevant']]
        answer_text = item['answer']['text']
        answer_start = item['answer']['start']
        
        converted_item = {
            'id': item['id'],
            'question': question,
            'context': context,
            'answers': {
                'text': [answer_text],
                'answer_start': [answer_start]
            }
        }
        converted_data.append(converted_item)
    
    return converted_data

# Load data
contexts, train_data, valid_data = load_data()
print(f"Loaded {len(contexts)} contexts")
print(f"Loaded {len(train_data)} training examples")
print(f"Loaded {len(valid_data)} validation examples")

# Combine train and valid for more training data
combined_train_data = train_data + valid_data
qa_train_data = prepare_qa_data(contexts, combined_train_data)
print(f"Combined training data: {len(qa_train_data)} examples")

# =============================================================================
# CELL 3: Enhanced Span Model Training with Epoch Checkpoints
# =============================================================================

def prepare_train_features_enhanced(examples, tokenizer, max_seq_length=384, doc_stride=128):
    """Enhanced training features for QA"""
    examples["question"] = [q.lstrip() for q in examples["question"]]
    
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id) if tokenizer.cls_token_id in input_ids else 0
        
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
    
    return tokenized_examples

def train_span_model_with_checkpoints():
    """Train span model with epoch-level checkpointing"""
    print("Starting enhanced span model training with epoch checkpoints...")
    
    # Enhanced hyperparameters
    model_name = "hfl/chinese-bert-wwm-ext"  # Better Chinese BERT
    max_seq_length = 384
    doc_stride = 128
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 1
    learning_rate = 2e-5
    num_train_epochs = 5  # More epochs with checkpointing
    
    # Initialize accelerator
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    
    # Set seed
    set_seed(42)
    
    # Load model and tokenizer
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config)
    
    # Preprocess dataset
    train_dataset = Dataset.from_list(qa_train_data)
    
    def prepare_train_features_wrapper(examples):
        return prepare_train_features_enhanced(examples, tokenizer, max_seq_length, doc_stride)
    
    train_dataset = train_dataset.map(
        prepare_train_features_wrapper,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    # Data collator and loader
    data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=per_device_train_batch_size
    )
    
    # Optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * max_train_steps),
        num_training_steps=max_train_steps,
    )
    
    # Prepare with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Training loop with epoch checkpointing
    progress_bar = tqdm(range(max_train_steps), desc="Training Enhanced Span Model")
    completed_steps = 0
    
    for epoch in range(num_train_epochs):
        print(f"\n=== Starting Epoch {epoch + 1}/{num_train_epochs} ===")
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
            
            if completed_steps >= max_train_steps:
                break
        
        avg_loss = total_loss.item() / len(train_dataloader)
        print(f"Epoch {epoch + 1} completed: Average loss = {avg_loss:.4f}")
        
        # CHECKPOINT: Save model after each epoch
        checkpoint_dir = f"span_model_epoch_{epoch + 1}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(checkpoint_dir, is_main_process=accelerator.is_main_process, 
                                       save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(checkpoint_dir)
        
        print(f"✓ Checkpoint saved: {checkpoint_dir}")
        
        if completed_steps >= max_train_steps:
            break
    
    # Final model save
    final_output_dir = "span_model_final"
    os.makedirs(final_output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(final_output_dir, is_main_process=accelerator.is_main_process, 
                                   save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(final_output_dir)
    
    print(f"✓ Final model saved: {final_output_dir}")
    print("Enhanced span model training completed with epoch checkpoints!")
    
    return unwrapped_model, tokenizer

# =============================================================================
# CELL 4: Train Enhanced Span Model
# =============================================================================

span_model, span_tokenizer = train_span_model_with_checkpoints()

# =============================================================================
# CELL 5: Summary and File Management
# =============================================================================

print("\n" + "="*80)
print("ENHANCED SPAN MODEL TRAINING COMPLETED!")
print("="*80)

print("\nModels saved:")
for epoch in range(1, 6):  # Adjust based on how many epochs completed
    checkpoint_dir = f"span_model_epoch_{epoch}"
    if os.path.exists(checkpoint_dir):
        print(f"✓ span_model_epoch_{epoch}/ - Model after epoch {epoch}")

if os.path.exists("span_model_final"):
    print("✓ span_model_final/ - Final trained model")

print("\nEnhancements:")
print("- Better Chinese BERT: hfl/chinese-bert-wwm-ext")
print("- Combined train+valid data: 24,723 examples")
print("- Optimized hyperparameters: lr=2e-5, weight_decay=0.01")
print("- Epoch-level checkpointing for safety")
print("- 5 epochs for better convergence")

print("\nNext steps:")
print("1. Download the models you want to test")
print("2. Use original paragraph model + new span models for inference")
print("3. Test different epoch checkpoints to find best performance")

print("\nExpected improvement: 0.73567 → 0.76+ (targeting strong baseline)")
print("="*80)