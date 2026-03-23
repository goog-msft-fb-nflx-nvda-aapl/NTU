import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm

from tokenization_qwen3 import Qwen3Tokenizer
from p2.dataset import CaptionDataset
from p2.model import LLaVACaptioner


def get_transform(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])


def train():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    is_main = (local_rank == 0)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'hw3_data', 'p2_data')
    train_image_dir = os.path.join(data_dir, 'images', 'train')
    train_ann = os.path.join(data_dir, 'train.json')
    decoder_path = os.path.join(data_dir, 'decoder_model.bin')
    save_dir = os.path.join(base_dir, 'p2', 'checkpoints_exp3')
    if is_main:
        os.makedirs(save_dir, exist_ok=True)

    vocab_file = os.path.join(base_dir, 'vocab.json')
    merges_file = os.path.join(base_dir, 'merges.txt')

    # Experiment 3: higher rank
    batch_size = 32
    num_epochs = 10
    lr = 2e-4
    max_length = 64
    lora_r = 16
    lora_alpha = 16

    tokenizer = Qwen3Tokenizer(vocab_file=vocab_file, merges_file=merges_file)

    train_dataset = CaptionDataset(
        image_dir=train_image_dir,
        annotation_file=train_ann,
        tokenizer=tokenizer,
        transform=get_transform(train=True),
        max_length=max_length,
    )
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=4, pin_memory=True
    )

    model = LLaVACaptioner(decoder_path=decoder_path, lora_r=lora_r, lora_alpha=lora_alpha)
    model = model.to(device)

    if is_main:
        n_params = model.count_trainable_params()
        print(f"Trainable parameters: {n_params:,}")

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        sampler.set_epoch(epoch)
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') if is_main else train_loader
        for images, input_ids in pbar:
            images = images.to(device)
            input_ids = input_ids.to(device)

            optimizer.zero_grad()
            loss = model(images, input_ids)
            loss.backward()
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0
            )
            optimizer.step()

            total_loss += loss.item()
            if is_main:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()

        if is_main:
            print(f'Epoch {epoch+1}: avg_loss={avg_loss:.4f}')

        scheduler.step()

        if is_main:
            raw_model = model.module
            trainable_names = {n for n, p in raw_model.named_parameters() if p.requires_grad}
            ckpt = {k: v for k, v in raw_model.state_dict().items() if k in trainable_names}

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(ckpt, os.path.join(save_dir, 'best_model.bin'))
                print(f'  -> Saved best model (loss={best_loss:.4f})')

            torch.save(ckpt, os.path.join(save_dir, f'epoch_{epoch+1}.bin'))

    dist.destroy_process_group()
    if is_main:
        print('Training complete.')


if __name__ == '__main__':
    train()
