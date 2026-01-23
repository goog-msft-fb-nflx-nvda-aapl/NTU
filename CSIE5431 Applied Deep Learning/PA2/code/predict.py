"""
Inference script for Classical Chinese translation
This script will be called by run.sh
"""
import os
import sys

# Add parent directory (root) to path to import utils
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
import json
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import get_prompt, get_bnb_config


def generate_predictions(model, tokenizer, test_data, max_new_tokens=128):
    """Generate predictions for test data."""
    model.eval()
    predictions = []
    
    for item in tqdm(test_data, desc="Generating predictions"):
        instruction = item['instruction']
        prompt = get_prompt(instruction)
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response (after "ASSISTANT:")
        if "ASSISTANT:" in full_response:
            response = full_response.split("ASSISTANT:")[-1].strip()
        else:
            response = full_response.strip()
        
        # Create prediction entry
        prediction = {
            'id': item['id'],
            'output': response
        }
        predictions.append(prediction)
    
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions for Classical Chinese translation"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to the base model (e.g., 'Qwen/Qwen3-4B' or local path)"
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        required=True,
        help="Path to the PEFT adapter checkpoint"
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="Path to test data (input JSON file)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save predictions (output JSON file)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate"
    )
    args = parser.parse_args()
    
    print("="*70)
    print("Classical Chinese Translation - Inference")
    print("="*70)
    print(f"Base Model: {args.base_model_path}")
    print(f"Adapter Path: {args.peft_path}")
    print(f"Test Data: {args.test_data_path}")
    print(f"Output Path: {args.output_path}")
    print(f"Max New Tokens: {args.max_new_tokens}")
    print("="*70)
    
    # Check if files exist
    if not os.path.exists(args.test_data_path):
        print(f"Error: Test data file not found: {args.test_data_path}")
        sys.exit(1)
    
    if not os.path.exists(args.peft_path):
        print(f"Error: PEFT adapter path not found: {args.peft_path}")
        sys.exit(1)
    
    print("\n[1/5] Loading tokenizer...")
    
    # Load tokenizer first (faster, helps verify model path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_path,
            trust_remote_code=True
        )
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        sys.exit(1)
    
    # Set special tokens
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
    
    print("\n[2/5] Loading base model...")
    
    # Get BnB config
    bnb_config = get_bnb_config()
    
    # Load base model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("✓ Base model loaded")
    except Exception as e:
        print(f"✗ Error loading base model: {e}")
        sys.exit(1)
    
    print("\n[3/5] Loading PEFT adapter...")
    
    # Load PEFT adapter
    try:
        model = PeftModel.from_pretrained(model, args.peft_path)
        print("✓ PEFT adapter loaded")
    except Exception as e:
        print(f"✗ Error loading PEFT adapter: {e}")
        sys.exit(1)
    
    print("\n[4/5] Loading test data...")
    
    # Load test data
    try:
        with open(args.test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"✓ Loaded {len(test_data)} test samples")
    except Exception as e:
        print(f"✗ Error loading test data: {e}")
        sys.exit(1)
    
    print("\n[5/5] Generating predictions...")
    print("="*70)
    
    # Generate predictions
    try:
        predictions = generate_predictions(
            model, 
            tokenizer, 
            test_data, 
            max_new_tokens=args.max_new_tokens
        )
        print(f"\n✓ Generated {len(predictions)} predictions")
    except Exception as e:
        print(f"\n✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\nSaving predictions...")
    
    # Save predictions
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        
        # Get file size
        file_size = os.path.getsize(args.output_path) / 1024  # KB
        print(f"✓ Predictions saved to: {args.output_path}")
        print(f"  File size: {file_size:.2f} KB")
    except Exception as e:
        print(f"✗ Error saving predictions: {e}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("✓ Prediction complete!")
    print("="*70)


if __name__ == "__main__":
    main()