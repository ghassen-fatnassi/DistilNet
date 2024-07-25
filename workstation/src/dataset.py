import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image
from utils import  load_yaml,UnetMaskProcessor
from tqdm.rich import trange,tqdm
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
        self.images=[]
        self.masks= []
        self.maskpreprocessor=UnetMaskProcessor()
        self.load_data_into_memory()
 
    def __len__(self):
        return len(self.img_paths)//self.cfg['divide_by']
    def load_data_into_memory(self):
        for i in trange(len(self.img_paths)//self.cfg['divide_by'],desc="Loading data into memory"):
            img_path=self.img_paths[i]
            mask_path=self.mask_paths[i]
            image=Image.open(img_path).resize(
                (self.Unet_cfg['resize'][1],self.Unet_cfg['resize'][0]),resample=Image.BILINEAR)
            image=torch.from_numpy(np.array(image)/255).float().permute(2,0,1)
            mask=Image.open(mask_path).resize(
                (self.Unet_cfg['resize'][1],self.Unet_cfg['resize'][0]),resample=Image.NEAREST)
            mask=self.maskpreprocessor.one_hotting_mask(np.array(mask))
            mask=torch.from_numpy(mask)
            self.images.append(image)
            self.masks.append(mask)
 
    def __getitem__(self, index):
        return self.images[index],self.masks[index]
 