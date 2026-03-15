import torch
import glob
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import utils
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

from transformers import GPT2Config, GPT2LMHeadModel

X_LEN = 1024

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_path', type=str, default='../basic_event_dictionary.pkl')
    parser.add_argument('--data_dir', type=str, default='../data/Pop1K7/midi_analyzed')
    parser.add_argument('--ckp_folder', type=str, default='../checkpoints')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--mode', type=str, default='train', choices=['train','generate'])
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='../results/midi')
    parser.add_argument('--n_generate', type=int, default=20)
    parser.add_argument('--n_target_bar', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=1.2)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--prompt_path', type=str, default='')
    parser.add_argument('--prompt_bars', type=int, default=0)
    args = parser.parse_args()
    return args

opt = parse_opt()
event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))
VOCAB_SIZE = len(event2word)
print(f'Vocab size: {VOCAB_SIZE}')


class MidiDataset(Dataset):
    def __init__(self, midi_paths):
        self.x_len = X_LEN
        self.event2word = event2word
        self.segments = self.prepare_data(midi_paths)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        return self.segments[index]

    def extract_events(self, input_path):
        try:
            note_items, tempo_items = utils.read_items(input_path)
            if len(note_items) == 0:
                return []
            note_items = utils.quantize_items(note_items)
            max_time = note_items[-1].end
            items = tempo_items + note_items
            groups = utils.group_items(items, max_time)
            events = utils.item2event(groups)
            return events
        except Exception as e:
            return []

    def prepare_data(self, midi_paths):
        all_words = []
        for path in tqdm(midi_paths, desc='Tokenizing'):
            events = self.extract_events(path)
            if not events:
                continue
            words = []
            for event in events:
                e = '{}_{}'.format(event.name, event.value)
                if e in self.event2word:
                    words.append(self.event2word[e])
                else:
                    if event.name == 'Note Velocity':
                        words.append(self.event2word['Note Velocity_21'])
            if len(words) > self.x_len + 1:
                all_words.append(words)

        segments = []
        for words in all_words:
            for i in range(0, len(words) - self.x_len - 1, self.x_len // 2):
                x = words[i:i + self.x_len]
                y = words[i+1:i + self.x_len + 1]
                if len(x) == self.x_len and len(y) == self.x_len:
                    segments.append([x, y])
        segments = np.array(segments)
        print(f'Total segments: {len(segments)}')
        return segments


def build_model():
    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=X_LEN,
        n_embd=512,
        n_layer=12,
        n_head=8,
        n_inner=2048,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    model = GPT2LMHeadModel(config)
    return model


def temperature_sampling(logits, temperature, topk):
    logits = logits / temperature
    if topk > 0:
        top_values, _ = torch.topk(torch.tensor(logits), topk)
        min_val = top_values[-1].item()
        logits[logits < min_val] = -float('Inf')
    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()
    return np.random.choice(len(probs), p=probs)


def generate(n_target_bar=32, temperature=1.2, topk=5, output_path='out.mid',
             model_path='', prompt_path='', prompt_bars=0):
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = build_model().to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    with torch.no_grad():
        if prompt_path and prompt_bars > 0:
            # encode prompt
            try:
                note_items, tempo_items = utils.read_items(prompt_path)
                note_items = utils.quantize_items(note_items)
                # only keep first prompt_bars bars
                ticks_per_bar = 480 * 4
                max_time = ticks_per_bar * prompt_bars
                note_items = [n for n in note_items if n.start < max_time]
                tempo_items_filtered = [t for t in tempo_items if t.start < max_time]
                if not tempo_items_filtered:
                    tempo_items_filtered = [tempo_items[0]]
                items = tempo_items_filtered + note_items
                groups = utils.group_items(items, max_time)
                events = utils.item2event(groups)
                words = []
                for event in events:
                    e = '{}_{}'.format(event.name, event.value)
                    if e in event2word:
                        words.append(event2word[e])
                    else:
                        if event.name == 'Note Velocity':
                            words.append(event2word['Note Velocity_21'])
                print(f'Prompt tokens: {len(words)}')
            except Exception as e:
                print(f'Prompt error: {e}, using random start')
                prompt_path = ''

        if not prompt_path or not prompt_bars:
            tempo_classes = [v for k, v in event2word.items() if 'Tempo Class' in k]
            tempo_values = [v for k, v in event2word.items() if 'Tempo Value' in k]
            words = [event2word['Bar_None'],
                     event2word['Position_1/16'],
                     np.random.choice(tempo_classes),
                     np.random.choice(tempo_values)]

        current_generated_bar = 0
        bar_count_start = words.count(event2word['Bar_None'])

        # use a sliding window context
        max_ctx = X_LEN

        print(f'Generating {n_target_bar} bars...')
        while current_generated_bar < n_target_bar:
            context = words[-max_ctx:] if len(words) > max_ctx else words
            temp_x = torch.tensor([context], dtype=torch.long).to(device)
            outputs = model(temp_x)
            logits = outputs.logits[0, -1].cpu().numpy()
            word = temperature_sampling(logits, temperature, topk)
            words.append(word)
            if word == event2word['Bar_None']:
                current_generated_bar += 1
                if current_generated_bar % 4 == 0:
                    print(f'  {current_generated_bar}/{n_target_bar} bars')

        utils.write_midi(
            words=words,
            word2event=word2event,
            output_path=output_path,
            prompt_path=prompt_path if (prompt_path and prompt_bars > 0) else None
        )
        print(f'Saved: {output_path}')


def train():
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    train_list = glob.glob(os.path.join(opt.data_dir, "**/*.mid"), recursive=True)
    print(f'Train files: {len(train_list)}')

    train_dataset = MidiDataset(train_list)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)

    model = build_model().to(device)
    # multi-gpu
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)

    os.makedirs(opt.ckp_folder, exist_ok=True)
    losses = []
    start_epoch = 1

    # resume if checkpoint exists
    latest = sorted(glob.glob(os.path.join(opt.ckp_folder, 'epoch_*.pkl')))
    if latest:
        ckpt = torch.load(latest[-1], map_location=device, weights_only=False)
        raw_model = model.module if hasattr(model, 'module') else model
        raw_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        losses = list(ckpt.get('losses', []))
        print(f'Resumed from epoch {ckpt["epoch"]}')

    print('Start training')
    for epoch in range(start_epoch, opt.epochs + 1):
        model.train()
        epoch_losses = []
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch}'):
            x = batch[:, 0, :].to(device).long()
            y = batch[:, 1, :].to(device).long()
            outputs = model(x, labels=y)
            loss = outputs.loss.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            epoch_losses.append(loss.item())
        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f'Epoch {epoch}: loss={avg_loss:.4f}')

        raw_model = model.module if hasattr(model, 'module') else model
        torch.save({
            'epoch': epoch,
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'losses': losses,
        }, os.path.join(opt.ckp_folder, f'epoch_{epoch:03d}.pkl'))
        np.save(os.path.join(opt.ckp_folder, 'training_loss.npy'), np.array(losses))


def main():
    if opt.mode == 'train':
        train()
    elif opt.mode == 'generate':
        os.makedirs(opt.output_dir, exist_ok=True)
        if opt.prompt_path:
            # Task 2: continuation
            out = os.path.join(opt.output_dir, f'continuation_t{opt.temperature}_k{opt.topk}.mid')
            generate(n_target_bar=opt.n_target_bar, temperature=opt.temperature,
                     topk=opt.topk, output_path=out, model_path=opt.model_path,
                     prompt_path=opt.prompt_path, prompt_bars=opt.prompt_bars)
        else:
            # Task 1: unconditional
            for i in range(opt.n_generate):
                out = os.path.join(opt.output_dir, f'gen_{i+1:02d}_t{opt.temperature}_k{opt.topk}.mid')
                generate(n_target_bar=opt.n_target_bar, temperature=opt.temperature,
                         topk=opt.topk, output_path=out, model_path=opt.model_path)


if __name__ == '__main__':
    main()
