import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, SGD, AdamW, Adamax
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from accelerate import Accelerator
import safetensors.torch
import wandb

import datetime
import os
import json

from .... import dataset,utils,loss
from ..Unet import segUnet
from .teacher_engine import engine

torch.manual_seed(50)

# Load configurations
cfg = utils.load_yaml()
Unet_cfg = utils.load_yaml(cfg['paths']['cfg']['Unet'])

# Set environment variables
os.environ['WANDB_API_KEY'] = cfg['wandb']['API_KEY']
os.environ["WANDB_SILENT"] = cfg['wandb']['silent']
os.environ["WANDB_DIR"] = f"{Unet_cfg['training']['log_dir']}/{Unet_cfg['model']}"

def generate_timestamp_id():
    # Get the current time
    now = datetime.datetime.now()
    # Format the time as a string: YYYYMMDD_HHMMSS
    timestamp_id = now.strftime("%Y%m%d_%H%M%S")
    return timestamp_id
identity=generate_timestamp_id()


# Dataset and DataLoaders configuration
batch_size = Unet_cfg['training']['batch_size']
data = dataset.SegDataset()
train_loader, val_loader = utils.datasetSplitter(data, batch_size).split() # if i wanna change the split , i just gotta change random seed in here

# Model configuration
num_classes = cfg['dataset']['num_classes']
in_channels = Unet_cfg['training']['in_channels']
depth = Unet_cfg['training']['depth']
start_filts = Unet_cfg['training']['start_filts']
teacher = segUnet(num_classes=num_classes, in_channels=in_channels, depth=depth, start_filts=start_filts,negative_slope=0.01)

# Optimizer
lr = Unet_cfg['training']['lr']
optimizer = AdamW(teacher.parameters(), lr=lr)

# Scheduler
factor = Unet_cfg['training']['factor']
patience = Unet_cfg['training']['patience']
scheduler = ReduceLROnPlateau(optimizer, factor=factor, patience=patience)

# Loss function
criterion = loss.WeightedCELoss()
if __name__ == '__main__':

    # Accelerator setup
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(project_name="ACTIA", config={'training':Unet_cfg['training'],'id':identity})
    accelerator.trackers[0].run.name = f'method=basicKD_training_id={identity}'
    # Prepare model and data for accelerator
    teacher, optimizer, criterion, scheduler, train_loader, val_loader = accelerator.prepare(
        teacher, optimizer, criterion, scheduler, train_loader, val_loader
    )

# Training the model
    engine(teacher, train_loader, val_loader, criterion, optimizer, scheduler, accelerator,epochs=Unet_cfg['training']['epochs'],img_sampling_index=9)
    accelerator.wait_for_everyone()



    """saving the model"""
    if(Unet_cfg['training']['save']):
        name =f"{Unet_cfg['training']['save_dir']}/{Unet_cfg['model']}/{identity}.safetensors"
        print("teacher model saved to : {name}")
        unwrapped_teacher = accelerator.unwrap_model(teacher)
        safetensors.torch.save_file(unwrapped_teacher.state_dict(), name)

