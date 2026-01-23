"""
QLoRA Fine-tuning Script for Qwen3-4B
Classical Chinese Instruction Tuning
"""

import torch
import json
import numpy as np
import os
import warnings
import argparse
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

from datasets import Dataset

# ============================================================================
# Configuration
# ============================================================================
class Config:
    """Training configuration"""
    # Model configuration
    BASE_MODEL = "Qwen/Qwen3-4B"
    OUTPUT_DIR = "./qlora_adapter"
    CHECKPOINT_DIR = "./checkpoints"
    
    # Training hyperparameters
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    MAX_LENGTH = 512
    WARMUP_STEPS = 100
    LOGGING_STEPS = 10
    SAVE_STEPS = 500
    
    # LoRA configuration
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ]

# ============================================================================
# Utility Functions
# ============================================================================
def get_prompt(instruction: str) -> str:
    """Format the instruction as a prompt for LLM."""
    return f"你是人工智慧助理,以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"

def get_bnb_config() -> BitsAndBytesConfig:
    """Get the BitsAndBytesConfig for 4-bit quantization."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

def load_data(file_path):
    """Load training data from JSON file."""
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} samples")
    return data

def prepare_dataset(data, tokenizer, max_length=512):
    """Prepare dataset for training."""
    print("Preparing dataset...")
    formatted_data = []
    
    for item in tqdm(data, desc="Formatting data"):
        instruction = item['instruction']
        output = item['output']
        
        # Format prompt
        prompt = get_prompt(instruction)
        full_text = prompt + output + tokenizer.eos_token
        
        formatted_data.append({
            'text': full_text,
            'instruction': instruction,
            'output': output
        })
    
    return Dataset.from_list(formatted_data)

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the examples."""
    result = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors=None
    )
    result['labels'] = result['input_ids'].copy()
    return result

# ============================================================================
# Main Training Function
# ============================================================================
def main(args):
    """Main training function"""
    
    # Update config with command line arguments
    if args.train_file:
        train_file = args.train_file
    else:
        train_file = "./train.json"
    
    if args.output_dir:
        Config.OUTPUT_DIR = args.output_dir
    
    if args.num_epochs:
        Config.NUM_EPOCHS = args.num_epochs
    
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    
    if args.learning_rate:
        Config.LEARNING_RATE = args.learning_rate
    
    print("="*70)
    print("QLoRA Fine-tuning for Qwen3-4B")
    print("="*70)
    print(f"Base Model: {Config.BASE_MODEL}")
    print(f"Output Directory: {Config.OUTPUT_DIR}")
    print(f"Training File: {train_file}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Gradient Accumulation: {Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective Batch Size: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Max Length: {Config.MAX_LENGTH}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print("="*70)
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # ========================================================================
    # Load Model and Tokenizer
    # ========================================================================
    print("\n[1/6] Loading model and tokenizer...")
    
    # Get BnB config
    bnb_config = get_bnb_config()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        Config.BASE_MODEL,
        trust_remote_code=True
    )
    
    # Set special tokens
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        Config.BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    print("✓ Model and tokenizer loaded")
    
    # ========================================================================
    # Configure LoRA
    # ========================================================================
    print("\n[2/6] Configuring LoRA...")
    
    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        target_modules=Config.LORA_TARGET_MODULES,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("✓ LoRA configured")
    
    # ========================================================================
    # Prepare Training Data
    # ========================================================================
    print("\n[3/6] Preparing training data...")
    
    # Load training data
    train_data = load_data(train_file)
    
    # Create dataset
    train_dataset = prepare_dataset(train_data, tokenizer, Config.MAX_LENGTH)
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, Config.MAX_LENGTH),
        batched=True,
        remove_columns=['text', 'instruction', 'output'],
        desc="Tokenizing"
    )
    
    print(f"✓ Dataset prepared: {len(train_dataset)} samples")
    
    # ========================================================================
    # Training Arguments
    # ========================================================================
    print("\n[4/6] Configuring training arguments...")
    
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=Config.LEARNING_RATE,
        fp16=False,
        bf16=True,
        logging_steps=Config.LOGGING_STEPS,
        logging_first_step=True,
        save_steps=Config.SAVE_STEPS,
        save_total_limit=3,
        warmup_steps=Config.WARMUP_STEPS,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        report_to="none",
        disable_tqdm=False,
        log_level="info",
    )
    
    steps_per_epoch = len(train_dataset) // (Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS)
    total_steps = steps_per_epoch * Config.NUM_EPOCHS
    
    print(f"✓ Training arguments configured")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Estimated time: ~{total_steps * 7 / 60:.0f} minutes")
    
    # ========================================================================
    # Data Collator and Trainer
    # ========================================================================
    print("\n[5/6] Initializing trainer...")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    print("✓ Trainer initialized")
    
    # ========================================================================
    # Train Model
    # ========================================================================
    print("\n[6/6] Starting training...")
    print("="*70)
    
    model.config.use_cache = False
    
    try:
        train_result = trainer.train()
        
        print("\n" + "="*70)
        print("✓ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Final training loss: {train_result.training_loss:.4f}")
        
    except Exception as e:
        print(f"\n✗ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Try emergency save
        print("\nAttempting emergency save...")
        try:
            emergency_dir = Config.OUTPUT_DIR + "_emergency"
            model.save_pretrained(emergency_dir)
            tokenizer.save_pretrained(emergency_dir)
            print(f"✓ Emergency checkpoint saved to {emergency_dir}")
        except:
            print("✗ Emergency save failed")
        raise
    
    # ========================================================================
    # Save Model
    # ========================================================================
    print("\n" + "="*70)
    print("Saving model...")
    print("="*70)
    
    try:
        model.save_pretrained(Config.OUTPUT_DIR)
        tokenizer.save_pretrained(Config.OUTPUT_DIR)
        
        # Save training log
        with open(f"{Config.OUTPUT_DIR}/log_history.json", 'w') as f:
            json.dump(trainer.state.log_history, f, indent=2)
        
        print(f"✓ Model saved to: {Config.OUTPUT_DIR}")
        print(f"✓ Training log saved")
        
        # List what was saved
        print("\nSaved files:")
        for item in os.listdir(Config.OUTPUT_DIR):
            filepath = os.path.join(Config.OUTPUT_DIR, item)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath) / (1024*1024)
                print(f"  - {item:40s} ({size:.2f} MB)")
        
        # Create zip archive
        import shutil
        archive_name = 'adapter_checkpoint'
        shutil.make_archive(archive_name, 'zip', Config.OUTPUT_DIR)
        zip_size = os.path.getsize(f'{archive_name}.zip') / (1024*1024)
        print(f"\n✓ Created {archive_name}.zip ({zip_size:.2f} MB)")
        print(f"  Upload this file to Google Drive for download.sh")
            
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n" + "="*70)
    print("✓ ALL DONE!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Upload {archive_name}.zip to Google Drive")
    print(f"2. Get a shareable link and update download.sh")
    print(f"3. Test inference with: bash run.sh")

# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning for Qwen3-4B")
    parser.add_argument("--train_file", type=str, default="./train.json",
                        help="Path to training data JSON file")
    parser.add_argument("--output_dir", type=str, default="./qlora_adapter",
                        help="Directory to save the trained adapter")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    
    args = parser.parse_args()
    
    main(args)