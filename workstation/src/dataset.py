import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image
from .utils import cityscapesMaskProcessor,bdd10kMaskProcessor,load_yaml

torch.manual_seed(50)

class SegDataset(Dataset):
    def __init__(self, config=None):# mode is either train_val or test
        super().__init__()
        if config is None:
            config = load_yaml()
        self.cfg = config
        self.Unet_cfg = load_yaml(self.cfg['paths']['cfg']['Unet'])

        self.img_dir = Path(self.cfg['dataset'][f'train_imgs_dir'])
        self.mask_dir = Path(self.cfg['dataset'][f'train_masks_dir'])

        self.img_paths = sorted(list(self.img_dir.glob('*.'+self.cfg['dataset']['img_type'])))
        self.mask_paths = sorted(list(self.mask_dir.glob('*.'+self.cfg['dataset']['mask_type'])))

        assert len(self.img_paths) == len(self.mask_paths), f"Number of images and masks do not match\  {len(self.img_paths)} does not match {len(self.mask_paths)} "

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]

        maskprocessor=bdd10kMaskProcessor()

        image = Image.open(img_path).resize((self.Unet_cfg['resize'][1], self.Unet_cfg['resize'][0]), resample=Image.BILINEAR)
        image = torch.from_numpy(np.array(image) / 255).float().permute(2, 0, 1)  # Normalize and permute to [C, H, W]

        mask = Image.open(mask_path).resize((self.Unet_cfg['resize'][1], self.Unet_cfg['resize'][0]), resample=Image.NEAREST)
        mask=maskprocessor.one_hotting_mask(np.array(mask))
        mask = torch.from_numpy(mask)

        return image, mask
