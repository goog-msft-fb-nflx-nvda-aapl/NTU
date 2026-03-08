"""
Generate two report figures:
  1. report_grid.png   : 10x10 grid (rows=digits, cols=10 samples)
  2. report_reverse.png: reverse process for digit 0 and digit 1
"""
import os
import numpy as np
from PIL import Image

OUTPUT_DIR = 'output_p1'
VIS_DIR    = os.path.join(OUTPUT_DIR, 'reverse_process')

# ─── Figure 1: 10x10 grid ────────────────────────────────────────────────────
# rows = digits 0-9, cols = first 10 samples
IMG_SIZE = 28
PAD = 2
N_COLS = 10
N_ROWS = 10

grid_h = N_ROWS * IMG_SIZE + (N_ROWS + 1) * PAD
grid_w = N_COLS * IMG_SIZE + (N_COLS + 1) * PAD
grid = Image.new('RGB', (grid_w, grid_h), (200, 200, 200))

for row, digit in enumerate(range(10)):
    if digit % 2 == 0:
        folder = os.path.join(OUTPUT_DIR, 'mnistm')
    else:
        folder = os.path.join(OUTPUT_DIR, 'svhn')

    for col in range(N_COLS):
        fname = f"{digit}_{col+1:03d}.png"
        img = Image.open(os.path.join(folder, fname)).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        x = PAD + col * (IMG_SIZE + PAD)
        y = PAD + row * (IMG_SIZE + PAD)
        grid.paste(img, (x, y))

grid.save('report_grid.png')
print("Saved report_grid.png")

# ─── Figure 2: Reverse process ───────────────────────────────────────────────
# rows = digit 0 and digit 1, cols = t=1000,800,600,400,200,1 (6 steps)
STEPS = [1000, 800, 600, 400, 200, 1]
N_R_ROWS = 2
N_R_COLS = len(STEPS)

r_grid_h = N_R_ROWS * IMG_SIZE + (N_R_ROWS + 1) * PAD
r_grid_w = N_R_COLS * IMG_SIZE + (N_R_COLS + 1) * PAD
r_grid = Image.new('RGB', (r_grid_w, r_grid_h), (200, 200, 200))

for row, digit in enumerate([0, 1]):
    for col, t_step in enumerate(STEPS):
        fname = f"digit{digit}_t{t_step:04d}.png"
        img = Image.open(os.path.join(VIS_DIR, fname)).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        x = PAD + col * (IMG_SIZE + PAD)
        y = PAD + row * (IMG_SIZE + PAD)
        r_grid.paste(img, (x, y))

r_grid.save('report_reverse.png')
print("Saved report_reverse.png")