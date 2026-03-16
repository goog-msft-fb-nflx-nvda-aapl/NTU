#!/bin/bash

# Download DeepLabV3+ (Model B / train_c) best checkpoint
gdown --id PLACEHOLDER_DEEPLAB_FILE_ID -O checkpoints/deeplab101_best.pth

# Download PSPNet (train_d) best checkpoint
gdown --id PLACEHOLDER_PSP_FILE_ID -O checkpoints_psp/pspnet_best.pth