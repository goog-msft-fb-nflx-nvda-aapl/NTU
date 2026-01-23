"""
Zero-Shot, Few-Shot, and LoRA Comparison Script
Analyze different inference strategies for Classical Chinese translation
"""

import os
import sys

# Add parent directory to path for utils import
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import get_prompt, get_bnb_config


# ============================================================================
# Utility Functions
# ============================================================================

def load_data(file_path):
    """Load data from JSON file."""
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} samples")
    return data


def generate_response(model, tokenizer, instruction, max_tokens=128, temperature=0.7):
    """Generate a response from the model."""
    prompt = get_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "ASSISTANT:" in response:
        response = response.split("ASSISTANT:")[-1].strip()
    return response


def create_fewshot_prompt(instruction: str, examples: list, num_examples: int = 5) -> str:
    """Create a few-shot prompt with in-context examples."""
    prompt = "你是文言古文翻譯助理。以下是一些例子：\n\n"
    
    # Add examples
    for i, example in enumerate(examples[:num_examples], 1):
        prompt += f"例子 {i}:\n"
        prompt += f"輸入: {example['instruction']}\n"
        prompt += f"輸出: {example['output']}\n\n"
    
    prompt += f"現在輪到你了：\n"
    prompt += f"輸入: {instruction}\n"
    prompt += f"輸出:"
    
    return prompt


def generate_fewshot_response(model, tokenizer, instruction, examples, 
                              num_examples=5, max_tokens=128, temperature=0.7):
    """Generate response using few-shot prompting."""
    prompt = create_fewshot_prompt(instruction, examples, num_examples)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the generated part after "輸出:"
    if "輸出:" in response:
        response = response.split("輸出:")[-1].strip()
    return response


def perplexity(model, tokenizer, data, max_length=2048):
    """Calculate perplexity for a dataset."""
    data_size = len(data)
    instructions = [get_prompt(x["instruction"]) for x in data]
    outputs = [x["output"] for x in data]

    # Tokenize data
    tokenized_instructions = tokenizer(instructions, add_special_tokens=False)
    tokenized_outputs = tokenizer(outputs, add_special_tokens=False)
    output_masks = []

    # Format data
    for i in range(data_size):
        instruction_input_ids = [tokenizer.bos_token_id] + tokenized_instructions["input_ids"][i]
        output_input_ids = tokenized_outputs["input_ids"][i] + [tokenizer.eos_token_id]
        tokenized_instructions["input_ids"][i] = instruction_input_ids + output_input_ids
        tokenized_instructions["attention_mask"][i] = [1] * len(tokenized_instructions["input_ids"][i])
        output_mask = [0] * len(instruction_input_ids) + [1] * len(output_input_ids)

        tokenized_instructions["input_ids"][i] = torch.tensor(
            tokenized_instructions["input_ids"][i][:max_length]
        )
        tokenized_instructions["attention_mask"][i] = torch.tensor(
            tokenized_instructions["attention_mask"][i][:max_length]
        )
        output_mask = torch.tensor(output_mask[:max_length])
        output_masks.append(output_mask)

    # Calculate ppl
    ppls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for i in tqdm(range(data_size), desc="Calculating perplexity"):
        input_ids = tokenized_instructions["input_ids"][i].unsqueeze(0).to(model.device)
        attn_mask = tokenized_instructions["attention_mask"][i].unsqueeze(0).to(model.device)
        output_mask = output_masks[i].unsqueeze(0).to(model.device)
        label = input_ids

        with torch.no_grad():
            out_logits = model(input_ids, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_label = label[..., 1:].contiguous()
        shift_output_mask = output_mask[..., 1:].contiguous()
        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_label) * shift_output_mask).sum(1)
            / shift_output_mask.sum(1)
        )
        ppls += perplexity_batch.tolist()
    
    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


def compute_model_perplexity(model, tokenizer, dataset, sample_size=50):
    """Compute average perplexity for a subset of the dataset."""
    subset = dataset[:sample_size]
    ppl_result = perplexity(model, tokenizer, subset)
    return ppl_result["mean_perplexity"]


def simple_length_similarity(gen, exp):
    """Simple similarity metric based on length ratio."""
    if len(exp) == 0:
        return 0.0
    ratio = len(gen) / len(exp) if len(exp) > 0 else 0
    return min(1.0, ratio) if ratio <= 1 else 1.0 / ratio


# ============================================================================
# Main Function
# ============================================================================

def main(args):
    """Main analysis function"""
    
    print("="*80)
    print("ZERO-SHOT, FEW-SHOT, AND LoRA COMPARISON")
    print("="*80)
    print(f"Base Model: {args.base_model_path}")
    print(f"Test Data: {args.test_data_path}")
    print(f"Train Data: {args.train_data_path}")
    if args.lora_path:
        print(f"LoRA Adapter: {args.lora_path}")
    print(f"Few-shot Examples: {args.num_fewshot}")
    print(f"Test Samples: {args.num_test_samples}")
    print(f"Output Directory: {args.output_dir}")
    print("="*80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ========================================================================
    # Load Models
    # ========================================================================
    print("\n[1/7] Loading models and tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path, 
        trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
    
    # Load base model (for zero-shot and few-shot)
    print("  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    base_model.eval()
    print("  ✓ Base model loaded")
    
    # Load LoRA model if provided
    lora_model = None
    if args.lora_path:
        print("  Loading LoRA-tuned model...")
        lora_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            quantization_config=get_bnb_config(),
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        lora_model = PeftModel.from_pretrained(lora_model, args.lora_path)
        lora_model.eval()
        print("  ✓ LoRA model loaded")
    
    print("✓ Models loaded successfully\n")
    
    # ========================================================================
    # Load Data
    # ========================================================================
    print("[2/7] Loading data...")
    
    train_data = load_data(args.train_data_path)
    test_data = load_data(args.test_data_path)
    
    # Prepare few-shot examples and test subset
    fewshot_examples = train_data[:args.num_fewshot]
    test_subset = test_data[:args.num_test_samples]
    
    print(f"✓ Using {len(fewshot_examples)} few-shot examples")
    print(f"✓ Testing on {len(test_subset)} samples\n")
    
    # ========================================================================
    # Zero-Shot Inference
    # ========================================================================
    print("[3/7] Running zero-shot inference...")
    print("="*80)
    print("ZERO-SHOT INFERENCE")
    print("="*80)
    print("Setting: Generate responses using base model without any in-context examples")
    print(f"Prompt design: {get_prompt('...')[:100]}...\n")
    
    zero_shot_results = []
    for test_item in tqdm(test_subset, desc="Zero-shot"):
        instruction = test_item['instruction']
        response = generate_response(base_model, tokenizer, instruction)
        zero_shot_results.append({
            'id': test_item['id'],
            'instruction': instruction,
            'expected': test_item['output'],
            'generated': response
        })
    
    print(f"✓ Zero-shot inference completed on {len(zero_shot_results)} samples\n")
    
    # ========================================================================
    # Few-Shot Inference
    # ========================================================================
    print("[4/7] Running few-shot inference...")
    print("="*80)
    print("FEW-SHOT (IN-CONTEXT LEARNING) INFERENCE")
    print("="*80)
    print(f"Setting: Using {args.num_fewshot} in-context examples from training data")
    print(f"Example selection: First {args.num_fewshot} samples from training data")
    print("Prompt design: Include examples in prompt, then ask model to perform task\n")
    
    few_shot_results = []
    for test_item in tqdm(test_subset, desc="Few-shot"):
        instruction = test_item['instruction']
        response = generate_fewshot_response(
            base_model, tokenizer, instruction, 
            fewshot_examples, num_examples=args.num_fewshot
        )
        few_shot_results.append({
            'id': test_item['id'],
            'instruction': instruction,
            'expected': test_item['output'],
            'generated': response
        })
    
    print(f"✓ Few-shot inference completed on {len(few_shot_results)} samples\n")
    
    # ========================================================================
    # LoRA Inference
    # ========================================================================
    lora_results = []
    if lora_model:
        print("[5/7] Running LoRA inference...")
        print("="*80)
        print("LoRA-TUNED MODEL INFERENCE")
        print("="*80)
        print("Setting: Use fine-tuned LoRA adapter with base model")
        print("Prompt design: Same as zero-shot (fine-tuning embedded in weights)\n")
        
        for test_item in tqdm(test_subset, desc="LoRA"):
            instruction = test_item['instruction']
            response = generate_response(lora_model, tokenizer, instruction)
            lora_results.append({
                'id': test_item['id'],
                'instruction': instruction,
                'expected': test_item['output'],
                'generated': response
            })
        
        print(f"✓ LoRA inference completed on {len(lora_results)} samples\n")
    else:
        print("[5/7] Skipping LoRA inference (no adapter provided)\n")
    
    # ========================================================================
    # Qualitative Comparison
    # ========================================================================
    print("[6/7] Generating qualitative comparison...")
    print("="*80)
    print("QUALITATIVE COMPARISON - Sample Outputs")
    print("="*80)
    
    num_samples_to_show = min(3, len(test_subset))
    qualitative_comparison = []
    
    for idx in range(num_samples_to_show):
        print(f"\n{'='*80}")
        print(f"Sample {idx + 1}")
        print(f"{'='*80}")
        print(f"Instruction: {test_subset[idx]['instruction'][:100]}...")
        print(f"\nExpected Output: {test_subset[idx]['output']}")
        print(f"\n--- ZERO-SHOT ---")
        print(f"Output: {zero_shot_results[idx]['generated']}")
        print(f"\n--- FEW-SHOT ({args.num_fewshot} examples) ---")
        print(f"Output: {few_shot_results[idx]['generated']}")
        
        sample_comparison = {
            'id': test_subset[idx]['id'],
            'instruction': test_subset[idx]['instruction'],
            'expected': test_subset[idx]['output'],
            'zero_shot': zero_shot_results[idx]['generated'],
            'few_shot': few_shot_results[idx]['generated']
        }
        
        if lora_model:
            print(f"\n--- LoRA-TUNED ---")
            print(f"Output: {lora_results[idx]['generated']}")
            sample_comparison['lora'] = lora_results[idx]['generated']
        
        qualitative_comparison.append(sample_comparison)
    
    # ========================================================================
    # Quantitative Analysis
    # ========================================================================
    print("\n[7/7] Computing quantitative metrics...")
    print("="*80)
    print("QUANTITATIVE COMPARISON")
    print("="*80)
    
    # Calculate basic metrics
    metrics = {
        'zero-shot': {
            'avg_length': np.mean([len(r['generated']) for r in zero_shot_results]),
            'responses': zero_shot_results
        },
        'few-shot': {
            'avg_length': np.mean([len(r['generated']) for r in few_shot_results]),
            'responses': few_shot_results
        }
    }
    
    if lora_model:
        metrics['lora'] = {
            'avg_length': np.mean([len(r['generated']) for r in lora_results]),
            'responses': lora_results
        }
    
    # Display basic metrics
    print(f"\n{'Metric':<20} {'Zero-Shot':<20} {'Few-Shot':<20}", end="")
    if lora_model:
        print(f" {'LoRA':<20}")
    else:
        print()
    
    print("-" * (60 if not lora_model else 80))
    
    print(f"{'Avg Output Length':<20} {metrics['zero-shot']['avg_length']:<20.2f} "
          f"{metrics['few-shot']['avg_length']:<20.2f}", end="")
    if lora_model:
        print(f" {metrics['lora']['avg_length']:<20.2f}")
    else:
        print()
    
    # Calculate similarity to expected output
    print(f"{'Avg Sim to Expected':<20} ", end="")
    for method in (['zero-shot', 'few-shot', 'lora'] if lora_model else ['zero-shot', 'few-shot']):
        sim = np.mean([simple_length_similarity(r['generated'], r['expected']) 
                       for r in metrics[method]['responses']])
        print(f"{sim:<20.4f} ", end="")
    print()
    
    # Compute perplexity
    print("\n" + "="*80)
    print("PERPLEXITY COMPARISON")
    print("="*80)
    
    print("→ Evaluating Zero-Shot (Base Model)...")
    ppl_zero = compute_model_perplexity(base_model, tokenizer, test_subset, args.ppl_samples)
    metrics['zero-shot']['ppl'] = ppl_zero
    
    print("→ Evaluating Few-Shot (Base Model with Context)...")
    # Construct few-shot-like dataset
    fewshot_like_data = []
    for item in test_subset[:args.ppl_samples]:
        fewshot_like_data.append({
            "instruction": create_fewshot_prompt(item["instruction"], fewshot_examples, args.num_fewshot),
            "output": item["output"]
        })
    ppl_few = compute_model_perplexity(base_model, tokenizer, fewshot_like_data, args.ppl_samples)
    metrics['few-shot']['ppl'] = ppl_few
    
    if lora_model:
        print("→ Evaluating LoRA Model...")
        ppl_lora = compute_model_perplexity(lora_model, tokenizer, test_subset, args.ppl_samples)
        metrics['lora']['ppl'] = ppl_lora
    
    # Display perplexity results
    print(f"\n{'Model':<20} {'Avg Perplexity':<20}")
    print("-" * 40)
    print(f"{'Zero-Shot':<20} {ppl_zero:<20.4f}")
    print(f"{'Few-Shot':<20} {ppl_few:<20.4f}")
    if lora_model:
        print(f"{'LoRA':<20} {ppl_lora:<20.4f}")
    
    # ========================================================================
    # Save Results
    # ========================================================================
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    results_summary = {
        'zero_shot': zero_shot_results,
        'few_shot': few_shot_results,
        'qualitative_comparison': qualitative_comparison,
        'metrics': {
            'zero_shot': {
                'avg_length': metrics['zero-shot']['avg_length'],
                'perplexity': ppl_zero
            },
            'few_shot': {
                'avg_length': metrics['few-shot']['avg_length'],
                'perplexity': ppl_few
            }
        },
        'settings': {
            'base_model': args.base_model_path,
            'zero_shot_prompt': 'Standard prompt without examples',
            'few_shot_examples': args.num_fewshot,
            'few_shot_selection_method': 'First N samples from training data',
            'test_samples': len(test_subset)
        }
    }
    
    if lora_model:
        results_summary['lora'] = lora_results
        results_summary['metrics']['lora'] = {
            'avg_length': metrics['lora']['avg_length'],
            'perplexity': ppl_lora
        }
    
    results_path = os.path.join(args.output_dir, 'inference_comparison_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"✓ Results saved to {results_path}")
    
    # ========================================================================
    # Create Visualizations
    # ========================================================================
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    methods = ['zero-shot', 'few-shot']
    colors = ['#FF6B6B', '#4ECDC4']
    if lora_model:
        methods.append('lora')
        colors.append('#45B7D1')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Output length comparison
    lengths = [
        [len(r['generated']) for r in metrics[m]['responses']]
        for m in methods
    ]
    
    axes[0].boxplot(lengths, labels=methods)
    axes[0].set_ylabel('Output Length (characters)')
    axes[0].set_title('Output Length Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Average lengths
    avg_lengths = [np.mean(l) for l in lengths]
    axes[1].bar(methods, avg_lengths, color=colors)
    axes[1].set_ylabel('Average Output Length')
    axes[1].set_title('Average Output Length Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Similarity scores
    similarities = [
        np.mean([simple_length_similarity(r['generated'], r['expected']) 
                 for r in metrics[m]['responses']])
        for m in methods
    ]
    axes[2].bar(methods, similarities, color=colors)
    axes[2].set_ylabel('Similarity Score')
    axes[2].set_title('Output Length Similarity to Expected')
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    comparison_path = os.path.join(args.output_dir, 'inference_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {comparison_path}")
    plt.close()
    
    # Perplexity comparison plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ppl_values = [ppl_zero, ppl_few]
    ppl_labels = ['Zero-Shot', 'Few-Shot']
    if lora_model:
        ppl_values.append(ppl_lora)
        ppl_labels.append('LoRA')
    
    bars = ax.bar(ppl_labels, ppl_values, color=colors)
    ax.set_ylabel('Average Perplexity')
    ax.set_title('Perplexity Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, ppl_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    ppl_path = os.path.join(args.output_dir, 'ppl_comparison.png')
    plt.savefig(ppl_path, dpi=300, bbox_inches='tight')
    print(f"✓ Perplexity visualization saved to {ppl_path}")
    plt.close()
    
    # ========================================================================
    # Generate Summary Report
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    report = f"""
INFERENCE STRATEGIES COMPARISON REPORT
{'='*80}

EXPERIMENTAL SETUP:
  Base Model: {args.base_model_path}
  Test Samples: {len(test_subset)}
  Few-shot Examples: {args.num_fewshot}
  PPL Evaluation Samples: {args.ppl_samples}

{'='*80}
ZERO-SHOT LEARNING:
  - Prompt Design: Standard instruction-following prompt with no examples
  - Setting: Using base model with only the task description in prompt
  - Average Output Length: {metrics['zero-shot']['avg_length']:.2f} characters
  - Perplexity: {ppl_zero:.4f}
  - Characteristics: Model relies entirely on pre-training knowledge

{'='*80}
FEW-SHOT LEARNING (IN-CONTEXT):
  - Prompt Design: Include {args.num_fewshot} examples followed by test instruction
  - Example Selection: First {args.num_fewshot} samples from training data
  - Number of Examples: {args.num_fewshot}
  - Average Output Length: {metrics['few-shot']['avg_length']:.2f} characters
  - Perplexity: {ppl_few:.4f}
  - Characteristics: Model learns task from examples without weight updates
"""
    
    if lora_model:
        report += f"""
{'='*80}
LoRA-TUNED MODEL:
  - Prompt Design: Same as zero-shot (knowledge in adapter weights)
  - Setting: Using fine-tuned LoRA adapter on base model
  - Adapter Path: {args.lora_path}
  - Average Output Length: {metrics['lora']['avg_length']:.2f} characters
  - Perplexity: {ppl_lora:.4f}
  - Characteristics: Model weights adapted through training on domain data
"""
    
    report += f"""
{'='*80}
KEY FINDINGS:

1. Perplexity Analysis:
   - Zero-shot: {ppl_zero:.4f}
   - Few-shot: {ppl_few:.4f}"""
    
    if lora_model:
        report += f"""
   - LoRA: {ppl_lora:.4f}"""
    
    report += f"""
   
   Lower perplexity indicates better fit to target distribution.

2. Output Length Trends:
   - Zero-shot avg: {metrics['zero-shot']['avg_length']:.2f}
   - Few-shot avg: {metrics['few-shot']['avg_length']:.2f}"""
    
    if lora_model:
        report += f"""
   - LoRA avg: {metrics['lora']['avg_length']:.2f}"""
    
    report += f"""

3. Learning Mechanisms:
   - Zero-shot: Relies on pre-training knowledge only
   - Few-shot: Learns from in-context examples during inference"""
    
    if lora_model:
        report += f"""
   - LoRA: Knowledge embedded in adapter weights"""
    
    report += f"""

4. Computational Cost:
   - Zero-shot: Minimal (shortest prompts)
   - Few-shot: Moderate (longer prompts with examples)"""
    
    if lora_model:
        report += f"""
   - LoRA: High training cost (already paid), minimal inference cost"""
    
    report += f"""

{'='*80}
OUTPUT FILES:
  - {results_path}
  - {comparison_path}
  - {ppl_path}
  - {os.path.join(args.output_dir, 'analysis_report.txt')}

{'='*80}
"""
    
    print(report)
    
    report_path = os.path.join(args.output_dir, 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✓ Report saved to {report_path}")
    
    print("\n" + "="*80)
    print("✓ INFERENCE STRATEGIES COMPARISON COMPLETED")
    print("="*80)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Zero-Shot, Few-Shot, and LoRA inference strategies"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Path to the base model"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to the LoRA adapter checkpoint (optional)"
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="Path to training data JSON file"
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="Path to test data JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./analysis_results",
        help="Directory to save results and visualizations"
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=5,
        help="Number of few-shot examples to use"
    )
    parser.add_argument(
        "--num_test_samples",
        type=int,
        default=50,
        help="Number of test samples to evaluate"
    )
    parser.add_argument(
        "--ppl_samples",
        type=int,
        default=50,
        help="Number of samples for perplexity calculation"
    )
    
    args = parser.parse_args()
    
    main(args)