import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from tokenization_qwen3 import Qwen3Tokenizer
from p2.dataset import ValDataset
from p2.model import LLaVACaptioner


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=str, help='Path to folder containing test images')
    parser.add_argument('output_json', type=str, help='Path to output JSON file')
    parser.add_argument('decoder_path', type=str, help='Path to decoder_model.bin')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to trained checkpoint')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Tokenizer
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vocab_file = os.path.join(base_dir, 'vocab.json')
    merges_file = os.path.join(base_dir, 'merges.txt')
    tokenizer = Qwen3Tokenizer(vocab_file=vocab_file, merges_file=merges_file)

    # Model
    model = LLaVACaptioner(
        decoder_path=args.decoder_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    # Load checkpoint
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(state, strict=False)
        print(f'Loaded checkpoint from {args.checkpoint}')

    model = model.to(device)
    model.eval()

    # Dataset
    dataset = ValDataset(image_dir=args.image_dir, transform=get_transform())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    results = {}
    for images, filenames in tqdm(loader, desc='Generating captions'):
        images = images.to(device)
        captions = model.generate(images, tokenizer, max_new_tokens=args.max_new_tokens)
        for fname, cap in zip(filenames, captions):
            # Remove extension as required
            key = os.path.splitext(fname)[0]
            results[key] = cap

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved {len(results)} captions to {args.output_json}')


if __name__ == '__main__':
    main()
