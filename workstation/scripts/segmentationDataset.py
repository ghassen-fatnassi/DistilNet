import torch
from torch.utils.data import Dataset

import numpy as np
from pathlib import Path
from PIL import Image

from utils import cityscapesMaskProcessor
from config_loader import load_yaml

class SegDataset(Dataset):
    def __init__(self, config=None, mode='train'):# mode is either train_val or test
        super().__init__()
        if config is None:
            config = load_yaml()
        self.cfg = config
        self.mode = mode

        self.img_dir = Path(self.cfg['dataset'][f'{mode}_imgs_dir'])
        self.mask_dir = Path(self.cfg['dataset'][f'{mode}_masks_dir'])

        self.img_paths = sorted(list(self.img_dir.glob('*.'+self.cfg['dataset']['img_type'])))  # Adjust pattern if needed
        self.mask_paths = sorted(list(self.mask_dir.glob('*.'+self.cfg['dataset']['mask_type'])))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]

        maskprocessor=cityscapesMaskProcessor()

        image = Image.open(img_path).resize((self.cfg['resize']['W'], self.cfg['resize']['H']), resample=Image.BILINEAR)
        image = torch.from_numpy(np.array(image) / 255).float().permute(2, 0, 1)  # Normalize and permute to [C, H, W]

        mask = Image.open(mask_path).resize((self.cfg['resize']['W'], self.cfg['resize']['H']), resample=Image.NEAREST)
        mask=maskprocessor.process_png_mask(mask)
        mask = torch.from_numpy(mask)

        return image, mask
