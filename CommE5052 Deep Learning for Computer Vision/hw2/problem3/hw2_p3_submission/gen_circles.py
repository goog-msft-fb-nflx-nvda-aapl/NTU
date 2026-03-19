import numpy as np
from PIL import Image, ImageDraw
import os

os.makedirs('report_circles', exist_ok=True)

# Experiment 1: Two circles side by side, different sizes
img1 = Image.new('RGB', (512, 512), color=(255, 255, 255))
draw1 = ImageDraw.Draw(img1)
draw1.ellipse([80, 156, 280, 356], fill=(0, 0, 0))    # left circle
draw1.ellipse([320, 156, 430, 266], fill=(0, 0, 0))   # right circle (smaller)
img1.save('report_circles/two_circles_1.png')

# Experiment 2: Two circles overlapping
img2 = Image.new('RGB', (512, 512), color=(255, 255, 255))
draw2 = ImageDraw.Draw(img2)
draw2.ellipse([100, 150, 350, 400], fill=(0, 0, 0))   # large circle
draw2.ellipse([280, 200, 430, 350], fill=(0, 0, 0))   # overlapping circle
img2.save('report_circles/two_circles_2.png')

print("Saved two circle control images.")