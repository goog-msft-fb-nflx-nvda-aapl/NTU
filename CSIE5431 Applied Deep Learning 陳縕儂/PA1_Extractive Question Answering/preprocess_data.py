#!/usr/bin/env python3
"""
Data preprocessing script for Chinese Extractive QA
Converts raw data format to formats suitable for paragraph selection and span selection models
"""

import json
import os
import argparse
from typing import List, Dict


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data for Chinese QA")
    parser.add_argument("--context_file", type=str, required=True, help="Path to context.json")
    parser.add_argument("--train_file", type=str, required=True, help="Path to train.json")
    parser.add_argument("--valid_file", type=str, required=True, help="Path to valid.json")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test.json")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="Output directory")
    return parser.parse_args()


def load_data(context_file: str, train_file: str, valid_file: str, test_file: str):
    """Load all data files"""
    with open(context_file, 'r', encoding='utf-8') as f:
        contexts = json.load(f)
    
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(valid_file, 'r', encoding='utf-8') as f:
        valid_data = json.load(f)
    
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    return contexts, train_data, valid_data, test_data


def prepare_paragraph_selection_data(contexts: List[str], data: List[Dict]) -> List[Dict]:
    """
    Convert data to SWAG-like format for paragraph selection
    Format: {sent1: question, sent2: "", ending0-3: paragraphs, label: correct_choice}
    """
    converted_data = []
    
    for item in data:
        question = item['question']
        paragraph_ids = item['paragraphs']
        paragraphs = [contexts[pid] for pid in paragraph_ids]
        
        converted_item = {
            'sent1': question,
            'sent2': '',
            'ending0': paragraphs[0],
            'ending1': paragraphs[1],
            'ending2': paragraphs[2],
            'ending3': paragraphs[3],
            'id': item['id']
        }
        
        # Add label for training/validation data
        if 'relevant' in item:
            relevant_id = item['relevant']
            label = paragraph_ids.index(relevant_id)
            converted_item['label'] = label
        
        converted_data.append(converted_item)
    
    return converted_data


def prepare_qa_data(contexts: List[str], data: List[Dict]) -> List[Dict]:
    """
    Convert data to SQuAD-like format for extractive QA
    Format: {id, question, context, answers: {text: [], answer_start: []}}
    """
    converted_data = []
    
    for item in data:
        question = item['question']
        
        if 'relevant' in item:
            # Training/validation data
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
        else:
            # Test data (no answers)
            converted_item = {
                'id': item['id'],
                'question': question,
                'context': '',  # Will be filled during inference
                'answers': {
                    'text': [],
                    'answer_start': []
                }
            }
        
        converted_data.append(converted_item)
    
    return converted_data


def save_json(data: List[Dict], output_file: str):
    """Save data as JSON"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} examples to {output_file}")


def main():
    args = parse_args()
    
    # Load data
    print("Loading data...")
    contexts, train_data, valid_data, test_data = load_data(
        args.context_file, args.train_file, args.valid_file, args.test_file
    )
    
    print(f"Loaded:")
    print(f"  Contexts: {len(contexts)}")
    print(f"  Train: {len(train_data)}")
    print(f"  Valid: {len(valid_data)}")
    print(f"  Test: {len(test_data)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process paragraph selection data
    print("\nPreparing paragraph selection data...")
    para_train = prepare_paragraph_selection_data(contexts, train_data)
    para_valid = prepare_paragraph_selection_data(contexts, valid_data)
    para_test = prepare_paragraph_selection_data(contexts, test_data)
    
    save_json(para_train, os.path.join(args.output_dir, 'para_train.json'))
    save_json(para_valid, os.path.join(args.output_dir, 'para_valid.json'))
    save_json(para_test, os.path.join(args.output_dir, 'para_test.json'))
    
    # Process QA data
    print("\nPreparing QA (span selection) data...")
    qa_train = prepare_qa_data(contexts, train_data)
    qa_valid = prepare_qa_data(contexts, valid_data)
    qa_test = prepare_qa_data(contexts, test_data)
    
    save_json(qa_train, os.path.join(args.output_dir, 'qa_train.json'))
    save_json(qa_valid, os.path.join(args.output_dir, 'qa_valid.json'))
    save_json(qa_test, os.path.join(args.output_dir, 'qa_test.json'))
    
    # Also save contexts for reference
    save_json(contexts, os.path.join(args.output_dir, 'contexts.json'))
    
    print("\nData preprocessing completed!")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()