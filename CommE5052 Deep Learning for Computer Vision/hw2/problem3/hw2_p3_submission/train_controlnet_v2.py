import sys
sys.path.insert(0, './stable-diffusion')
sys.path.insert(0, '/home/jtan/ControlNet')

import json
import cv2
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import argparse
import os


class Fill50kDataset(Dataset):
    def __init__(self, data_root, split='training'):
        self.data = []
        prompt_file = os.path.join(data_root, split, 'prompt.json')
        self.split_dir = os.path.join(data_root, split)
        with open(prompt_file, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(os.path.join(self.split_dir, source_filename))
        target = cv2.imread(os.path.join(self.split_dir, target_filename))

        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='hw2_data/fill50k')
    parser.add_argument('--ini_ckpt', type=str, default='/home/jtan/ControlNet/models/control_sd15_ini.ckpt')
    parser.add_argument('--config', type=str, default='/home/jtan/ControlNet/models/cldm_v15.yaml')
    parser.add_argument('--save_dir', type=str, default='controlnet_ckpt')
    parser.add_argument('--max_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--logger_freq', type=int, default=300)
    opt = parser.parse_args()

    model = create_model(opt.config).cpu()
    model.load_state_dict(load_state_dict(opt.ini_ckpt, location='cpu'))
    model.learning_rate = opt.lr
    model.sd_locked = True
    model.only_mid_control = False

    dataset = Fill50kDataset(opt.data_root, split='training')
    dataloader = DataLoader(dataset, num_workers=4, batch_size=opt.batch_size, shuffle=True)

    logger = ImageLogger(batch_frequency=opt.logger_freq)

    os.makedirs(opt.save_dir, exist_ok=True)
    # trainer = pl.Trainer(
    #     gpus=1,
    #     precision=32,
    #     callbacks=[logger],
    #     default_root_dir=opt.save_dir,
    #     max_epochs=opt.max_epochs,
    # )
    trainer = pl.Trainer(
        gpus=8,
        accelerator='ddp',
        precision=32,
        callbacks=[logger],
        default_root_dir=opt.save_dir,
        max_epochs=opt.max_epochs,
    )

    trainer.fit(model, dataloader)