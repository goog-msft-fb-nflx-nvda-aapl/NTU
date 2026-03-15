# HW3: Symbolic Music Generation

## Environment Setup
```bash
conda activate rl_hw3
pip install -r requirements.txt
```

## Inference (Unconditional Generation - Task 1)
```bash
cd src
python main.py \
  --mode generate \
  --dict_path ../basic_event_dictionary.pkl \
  --model_path ../checkpoints/epoch_200.pkl \
  --output_dir ../results/midi/output \
  --n_generate 20 \
  --n_target_bar 32 \
  --temperature 1.2 \
  --topk 5
```

## Inference (Continuation - Task 2)
```bash
cd src
python main.py \
  --mode generate \
  --dict_path ../basic_event_dictionary.pkl \
  --model_path ../checkpoints/epoch_200.pkl \
  --output_dir ../results/midi/output \
  --n_target_bar 24 \
  --temperature 1.2 \
  --topk 5 \
  --prompt_path /path/to/prompt.mid \
  --prompt_bars 8
```

## Convert MIDI to WAV
```bash
fluidsynth -ni -F output.wav -r 44100 /path/to/FluidR3Mono_GM.sf3 output.mid
```

## Evaluation
```bash
cd src
PYTHONPATH=/path/to/MusDr python eval_metrics.py \
  --dict_path ../basic_event_dictionary.pkl \
  --output_file_path /path/to/midi/folder
```

## Training (optional, from scratch)
```bash
cd src
python main.py \
  --mode train \
  --dict_path ../basic_event_dictionary.pkl \
  --data_dir ../data/Pop1K7/midi_analyzed \
  --ckp_folder ../checkpoints \
  --device cuda \
  --batch_size 16 \
  --epochs 200 \
  --lr 2e-4
```

## Model
- Architecture: GPT-2 (12 layers, 8 heads, d_model=512)
- Representation: REMI
- Vocabulary size: 249
- Context length: 1024
