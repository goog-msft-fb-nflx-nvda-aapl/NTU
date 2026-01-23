"""
Evaluation and Visualization Script
Evaluate model on public test set and plot learning curves
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from utils import get_prompt, get_bnb_config


def calculate_perplexity(model, tokenizer, data, max_length=512):
    """Calculate perplexity on test data."""
    model.eval()
    data_size = len(data)
    ppls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    
    for item in tqdm(data, desc="Calculating perplexity"):
        instruction = item['instruction']
        output = item['output']
        
        # Format prompt
        prompt = get_prompt(instruction)
        full_text = prompt + output
        
        # Tokenize
        instruction_tokens = tokenizer(prompt, add_special_tokens=False)
        output_tokens = tokenizer(output, add_special_tokens=False)
        
        # Create input
        input_ids = [tokenizer.bos_token_id] + instruction_tokens['input_ids'] + \
                    output_tokens['input_ids'] + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        
        # Create output mask (only calculate loss on output tokens)
        output_mask = [0] * (len(instruction_tokens['input_ids']) + 1) + \
                      [1] * (len(output_tokens['input_ids']) + 1)
        
        # Truncate
        input_ids = torch.tensor(input_ids[:max_length]).unsqueeze(0).cuda()
        attention_mask = torch.tensor(attention_mask[:max_length]).unsqueeze(0).cuda()
        output_mask = torch.tensor(output_mask[:max_length]).unsqueeze(0).cuda()
        
        # Forward pass
        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits
        
        # Calculate perplexity
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = output_mask[..., 1:].contiguous()
        
        loss = loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_mask
        ppl = torch.exp(loss.sum() / shift_mask.sum())
        ppls.append(ppl.item())
    
    return {
        'perplexities': ppls,
        'mean_perplexity': np.mean(ppls),
        'median_perplexity': np.median(ppls),
        'std_perplexity': np.std(ppls)
    }


def plot_learning_curve(log_history, output_path='learning_curve.png'):
    """Plot training loss curve."""
    steps = []
    losses = []
    
    for entry in log_history:
        if 'loss' in entry:
            steps.append(entry['step'])
            losses.append(entry['loss'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, 'b-', linewidth=2)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Learning curve saved to {output_path}")


def plot_perplexity_over_checkpoints(checkpoint_ppls, output_path='ppl_curve.png'):
    """Plot perplexity over different checkpoints."""
    checkpoints = list(checkpoint_ppls.keys())
    ppls = list(checkpoint_ppls.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(checkpoints, ppls, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Checkpoint (Steps)', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)
    plt.title('Perplexity on Public Test Set', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on points
    for x, y in zip(checkpoints, ppls):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Perplexity curve saved to {output_path}")


def evaluate_model(base_model_path, peft_path, test_data_path):
    """Evaluate model and return metrics."""
    print("Loading model...")
    
    # Load model
    bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
    
    # Load PEFT
    model = PeftModel.from_pretrained(model, peft_path)
    
    # Load test data
    print("Loading test data...")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"Evaluating on {len(test_data)} samples...")
    results = calculate_perplexity(model, tokenizer, test_data)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Mean Perplexity: {results['mean_perplexity']:.4f}")
    print(f"Median Perplexity: {results['median_perplexity']:.4f}")
    print(f"Std Perplexity: {results['std_perplexity']:.4f}")
    print("="*50)
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--peft_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--log_history_path", type=str, default=None,
                        help="Path to training log history JSON file")
    parser.add_argument("--output_dir", type=str, default="./results")
    
    args = parser.parse_args()
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate model
    results = evaluate_model(
        args.base_model_path,
        args.peft_path,
        args.test_data_path
    )
    
    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")
    
    # Plot learning curve if log history provided
    if args.log_history_path and os.path.exists(args.log_history_path):
        with open(args.log_history_path, 'r') as f:
            log_history = json.load(f)
        
        plot_path = os.path.join(args.output_dir, "learning_curve.png")
        plot_learning_curve(log_history, plot_path)


if __name__ == "__main__":
    main()