#!/usr/bin/env python3
"""
Inference script for Chinese Extractive QA
Uses trained paragraph selection and span selection models
"""

import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator
)
from datasets import Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for Chinese QA")
    parser.add_argument("--context_file", type=str, required=True, help="Path to context.json")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test.json")
    parser.add_argument("--paragraph_model", type=str, required=True, help="Path to paragraph selection model")
    parser.add_argument("--span_model", type=str, required=True, help="Path to span selection model")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output prediction.csv")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    return parser.parse_args()


def load_data(context_file, test_file):
    """Load context and test data"""
    with open(context_file, 'r', encoding='utf-8') as f:
        contexts = json.load(f)
    
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    return contexts, test_data


def prepare_paragraph_selection_data(contexts, data):
    """Convert data to format for paragraph selection"""
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
        
        converted_data.append(converted_item)
    
    return converted_data


def preprocess_paragraph_function(examples, tokenizer, max_seq_length=512):
    """Preprocess for paragraph selection"""
    first_sentences = [[context] * 4 for context in examples["sent1"]]
    question_headers = examples["sent2"]
    second_sentences = [
        [f"{header} {examples[end][i]}" for end in ["ending0", "ending1", "ending2", "ending3"]] 
        for i, header in enumerate(question_headers)
    ]
    
    # Flatten
    first_sentences = list(chain(*first_sentences))
    second_sentences = list(chain(*second_sentences))
    
    # Tokenize
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        max_length=max_seq_length,
        padding=True,
        truncation=True,
    )
    
    # Un-flatten
    tokenized_inputs = {k: [v[i:i+4] for i in range(0, len(v), 4)] 
                       for k, v in tokenized_examples.items()}
    
    return tokenized_inputs


def predict_paragraphs(test_data, model, tokenizer, contexts, batch_size, device):
    """Predict relevant paragraphs"""
    print("Step 1: Predicting relevant paragraphs...")
    
    # Prepare test data
    para_test_processed = prepare_paragraph_selection_data(contexts, test_data)
    test_dataset = Dataset.from_list(para_test_processed)
    
    def preprocess_with_tokenizer(examples):
        return preprocess_paragraph_function(examples, tokenizer, 512)
    
    processed_dataset = test_dataset.map(
        preprocess_with_tokenizer,
        batched=True,
        remove_columns=test_dataset.column_names
    )
    
    test_dataloader = DataLoader(processed_dataset, batch_size=batch_size, collate_fn=default_data_collator)
    
    model.eval()
    model.to(device)
    
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Paragraph selection"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            batch_predictions = outputs.logits.argmax(dim=-1)
            predictions.extend(batch_predictions.cpu().numpy())
    
    return predictions


def prepare_qa_test_data(test_data, paragraph_predictions, contexts):
    """Prepare QA data using predicted paragraphs"""
    qa_test_data = []
    
    for i, item in enumerate(test_data):
        question = item['question']
        paragraph_ids = item['paragraphs']
        predicted_para_idx = paragraph_predictions[i]
        relevant_paragraph_id = paragraph_ids[predicted_para_idx]
        context = contexts[relevant_paragraph_id]
        
        qa_item = {
            'id': item['id'],
            'question': question,
            'context': context
        }
        qa_test_data.append(qa_item)
    
    return qa_test_data


def prepare_validation_features(examples, tokenizer, max_seq_length):
    """Prepare features for span prediction"""
    examples["question"] = [q.lstrip() for q in examples["question"]]
    
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []
    
    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])
        
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]
    
    return tokenized_examples


def postprocess_qa_predictions(features, examples, predictions, n_best_size=20, max_answer_length=30):
    """Post-process QA predictions"""
    start_logits, end_logits = predictions
    
    example_id_to_index = {example["id"]: i for i, example in enumerate(examples)}
    features_per_example = {}
    for i, feature in enumerate(features):
        example_id = feature["example_id"]
        if example_id not in features_per_example:
            features_per_example[example_id] = []
        features_per_example[example_id].append(i)
    
    predictions_dict = {}
    
    for example_id, example in zip([ex["id"] for ex in examples], examples):
        if example_id not in features_per_example:
            predictions_dict[example_id] = ""
            continue
            
        feature_indices = features_per_example[example_id]
        prelim_predictions = []
        
        for feature_index in feature_indices:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            
            start_indexes = np.argsort(start_logit)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best_size - 1 : -1].tolist()
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    prelim_predictions.append(
                        {
                            "offsets": (start_char, end_char),
                            "score": start_logit[start_index] + end_logit[end_index],
                        }
                    )
        
        predictions_list = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]
        
        context = example["context"]
        for pred in predictions_list:
            offsets = pred["offsets"]
            pred["text"] = context[offsets[0] : offsets[1]]
        
        if len(predictions_list) == 0:
            predictions_dict[example_id] = ""
        else:
            predictions_dict[example_id] = predictions_list[0]["text"]
    
    return predictions_dict


def predict_spans(qa_test_data, model, tokenizer, batch_size, device, max_seq_length):
    """Predict answer spans"""
    print("Step 2: Predicting answer spans...")
    
    test_dataset = Dataset.from_list(qa_test_data)
    
    def prepare_validation_features_wrapper(examples):
        return prepare_validation_features(examples, tokenizer, max_seq_length)
    
    processed_dataset = test_dataset.map(
        prepare_validation_features_wrapper,
        batched=True,
        remove_columns=test_dataset.column_names
    )
    
    dataset_for_model = processed_dataset.remove_columns(["example_id", "offset_mapping"])
    data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    test_dataloader = DataLoader(dataset_for_model, batch_size=batch_size, collate_fn=data_collator)
    
    model.eval()
    model.to(device)
    
    all_start_logits = []
    all_end_logits = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Span selection"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())
    
    start_logits = np.concatenate(all_start_logits, axis=0)
    end_logits = np.concatenate(all_end_logits, axis=0)
    
    answers = postprocess_qa_predictions(
        processed_dataset, qa_test_data, (start_logits, end_logits)
    )
    
    return answers


def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data...")
    contexts, test_data = load_data(args.context_file, args.test_file)
    print(f"Loaded {len(contexts)} contexts and {len(test_data)} test examples")
    
    # Load models
    print(f"Loading paragraph selection model from {args.paragraph_model}...")
    para_tokenizer = AutoTokenizer.from_pretrained(args.paragraph_model)
    para_model = AutoModelForMultipleChoice.from_pretrained(args.paragraph_model)
    
    print(f"Loading span selection model from {args.span_model}...")
    span_tokenizer = AutoTokenizer.from_pretrained(args.span_model)
    span_model = AutoModelForQuestionAnswering.from_pretrained(args.span_model)
    
    # Predict paragraphs
    paragraph_predictions = predict_paragraphs(
        test_data, para_model, para_tokenizer, contexts, args.batch_size, device
    )
    
    # Prepare QA data
    qa_test_data = prepare_qa_test_data(test_data, paragraph_predictions, contexts)
    
    # Predict spans (use 384 for enhanced model)
    answers = predict_spans(
        qa_test_data, span_model, span_tokenizer, args.batch_size, device, max_seq_length=384
    )
    
    # Create submission
    print(f"Creating prediction file...")
    submission_data = []
    for item in test_data:
        submission_data.append({
            'id': item['id'],
            'answer': answers.get(item['id'], '')
        })
    
    # Save results
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(args.output_file, index=False, encoding='utf-8')
    
    print(f"Predictions saved to {args.output_file}")
    print(f"Total predictions: {len(submission_data)}")
    print(f"Empty predictions: {(submission_df['answer'] == '').sum()}")


if __name__ == "__main__":
    main()
