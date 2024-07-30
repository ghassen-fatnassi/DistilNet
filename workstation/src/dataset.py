import torch
from torchvision.transforms import v2,InterpolationMode
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image
from utils import  load_yaml,UnetMaskProcessor
from tqdm.autonotebook import trange,tqdm
torch.manual_seed(50)

class SegDataset(Dataset):
    def __init__(self, config=None):# mode is either train_val or test
        super().__init__()
        if config is None:
            config = load_yaml()
        self.cfg = config
        self.Unet_cfg = load_yaml(self.cfg['paths']['cfg']['Unet'])
        self.D=self.Unet_cfg['resize'][1]
        self.img_dir = Path(self.cfg['dataset'][f'train_imgs_dir'])
        self.mask_dir = Path(self.cfg['dataset'][f'train_masks_dir'])
 
        self.img_paths = sorted(list(self.img_dir.glob('*.'+self.cfg['dataset']['img_type'])))
        self.mask_paths = sorted(list(self.mask_dir.glob('*.'+self.cfg['dataset']['mask_type'])))
 
        assert len(self.img_paths) == len(self.mask_paths), f"Number of images and masks do not match\  {len(self.img_paths)} does not match {len(self.mask_paths)} "
        self.maskpreprocessor=UnetMaskProcessor()

        self.T_for_both=v2.Compose([
            v2.RandomHorizontalFlip(p=0.3),
            v2.RandomVerticalFlip(p=0.3),
            v2.RandomRotation(degrees=40),
        ])
        self.T_for_mask=v2.Compose([
            v2.Resize((self.D,self.D),interpolation=InterpolationMode.NEAREST),
        ])
        self.T_for_image=v2.Compose([
            v2.Resize((self.D,self.D),interpolation=InterpolationMode.BILINEAR) 
        ])

    def __len__(self):
        return len(self.img_paths)//self.cfg['dataset']['divide']
    
    def __getitem__(self, index):
        torch.manual_seed(50)

        img_path=self.img_paths[index]
        mask_path=self.mask_paths[index]
        image=Image.open(img_path)
        mask=Image.open(mask_path)
        image,mask=self.T_for_both(image,mask)
        image=self.T_for_image(image)
        image=torch.from_numpy(np.array(image)/255).float().permute(2,0,1)
        mask=self.T_for_mask(mask)
        mask=torch.from_numpy(self.maskpreprocessor.one_hotting_mask(np.array(mask)))

        return image,mask
