import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, SGD, AdamW, Adamax
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from accelerate import Accelerator

from dataset import SegDataset
from utils import datasetSplitter, load_yaml
import loss as loss
from models.Unet import segUnet
from teacher_engine import engine
import wandb
import os
import json

# Load configurations
cfg = load_yaml()
Unet_cfg = load_yaml(cfg['paths']['cfg']['Unet'])

# Set environment variables
os.environ['WANDB_API_KEY'] = cfg['wandb']['API_KEY']
os.environ["WANDB_SILENT"] = cfg['wandb']['silent']
os.environ["WANDB_DIR"] = Unet_cfg['training']['log_dir']

# Dataset and DataLoaders configuration
batch_size = Unet_cfg['training']['batch_size']
data = SegDataset()
train_loader, val_loader = datasetSplitter(data, batch_size).split() # if i wanna change the split , i just gotta change random seed in here

# Model configuration
num_classes = cfg['dataset']['num_classes']
in_channels = Unet_cfg['training']['in_channels']
depth = Unet_cfg['training']['depth']
start_filts = Unet_cfg['training']['start_filts']
teacher = segUnet(num_classes=num_classes, in_channels=in_channels, depth=depth, start_filts=start_filts)

# Optimizer
lr = Unet_cfg['training']['lr']
optimizer = AdamW(teacher.parameters(), lr=lr)

# Scheduler
factor = Unet_cfg['training']['factor']
patience = Unet_cfg['training']['patience']
scheduler = ReduceLROnPlateau(optimizer, factor=factor, patience=patience)

# Loss function
criterion = loss.WeightedCELoss()

# Accelerator setup
accelerator = Accelerator(log_with="wandb")
accelerator.init_trackers(project_name="ACTIA", config=Unet_cfg['training'])

# Prepare model and data for accelerator
teacher, optimizer, criterion, scheduler, train_loader, val_loader = accelerator.prepare(
    teacher, optimizer, criterion, scheduler, train_loader, val_loader
)

# Training the model
if __name__ == '__main__':
    engine(teacher, train_loader, val_loader, criterion, optimizer, scheduler, accelerator,epochs=Unet_cfg['training']['epochs'],img_sampling_index=9)
    accelerator.wait_for_everyone()

"""saving the model"""
if(Unet_cfg['training']['save']):
    accelerator.save(teacher, f"TEACHER_UNET::::{Unet_cfg['training']['save_dir']}::::{json.dumps(Unet_cfg['training'])}.pth")
    