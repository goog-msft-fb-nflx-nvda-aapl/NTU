#!/bin/bash
python3 src/inference.py $1 $2 --ckpt_deeplab checkpoints/deeplab101_best.pth --ckpt_psp checkpoints_psp/pspnet_best.pth