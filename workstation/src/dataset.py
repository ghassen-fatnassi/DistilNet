import torch
from torchvision.transforms import v2, InterpolationMode
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image
from utils import load_yaml, UnetMaskProcessor

class SegDataset(Dataset):
    def __init__(self, augment=False, config=None):
        super().__init__()
        if config is None:
            config = load_yaml()
        self.augment = augment
        self.cfg = config
        self.Unet_cfg = load_yaml(self.cfg['paths']['cfg']['Unet'])
        self.D = self.Unet_cfg['resize'][1]
        self.img_dir = Path(self.cfg['dataset']['train_imgs_dir'])
        self.mask_dir = Path(self.cfg['dataset']['train_masks_dir'])
        self.img_paths = sorted(list(self.img_dir.glob('*.' + self.cfg['dataset']['img_type'])))
        self.mask_paths = sorted(list(self.mask_dir.glob('*.' + self.cfg['dataset']['mask_type'])))
        assert len(self.img_paths) == len(self.mask_paths)
        
        self.maskpreprocessor = UnetMaskProcessor()

        self.T_for_both = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([v2.RandomRotation(degrees=60),], p=0.5),
            v2.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.9, 1.1)),
        ])
        self.T_image_only = v2.Compose([
            v2.RandomPhotometricDistort(p=0.3),
            v2.RandomGrayscale(p=0.5),
        ])
        self.T_for_mask = v2.Compose([
            v2.Resize((self.D, self.D), interpolation=InterpolationMode.NEAREST),
        ])
        self.T_for_image = v2.Compose([
            v2.Resize((self.D, self.D), interpolation=InterpolationMode.BILINEAR),
        ])

    def __len__(self):
        return len(self.img_paths) // self.cfg['dataset']['divide']

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        mask = Image.open(self.mask_paths[index])
        if self.augment:
            image, mask = self.T_for_both(image, mask)
            image = self.T_image_only(image)
        image = self.T_for_image(image)
        image = torch.from_numpy(np.array(image) / 255).float().permute(2, 0, 1)
        mask = self.T_for_mask(mask)
        mask = torch.from_numpy(self.maskpreprocessor.one_hotting_mask(np.array(mask)))
        return image, mask
