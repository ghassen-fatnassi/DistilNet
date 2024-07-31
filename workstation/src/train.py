import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, SGD, AdamW, Adamax
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau,CosineAnnealingWarmRestarts,StepLR,SequentialLR
from accelerate import Accelerator
import safetensors.torch
import wandb
import logging
import datetime
import os
import json

import dataset,utils,loss
from Unet import Unet,Res152Unet,Res34Unet
from teacher_engine import engine

torch.manual_seed(50)

# Load configurations
cfg = utils.load_yaml()
Unet_cfg = utils.load_yaml(cfg['paths']['cfg']['Unet'])

# Set environment variables
os.environ['WANDB_API_KEY'] = cfg['wandb']['API_KEY']
os.environ["WANDB_SILENT"] = cfg['wandb']['silent']
os.environ["WANDB_DIR"] = f"{Unet_cfg['teacher']['log_dir']}/Unet"
wandb.util.logger.setLevel(logging.ERROR)


def generate_timestamp_id():
    # Get the current time
    now = datetime.datetime.now()
    # Format the time as a string: YYYYMMDD_HHMMSS
    timestamp_id = now.strftime("%Y%m%d_%H%M%S")
    return timestamp_id
identity=generate_timestamp_id()


# Dataset and DataLoaders configuration
batch_size = Unet_cfg['teacher']['batch_size']
data = dataset.SegDataset()
train_loader, val_loader = utils.datasetSplitter(data, batch_size).split() # if i wanna change the split , i just gotta change random seed in here

# Model configuration
num_classes = cfg['dataset']['num_classes']
in_channels = Unet_cfg['in_channels']
depth = Unet_cfg['teacher']['depth']
start_filts = Unet_cfg['teacher']['start_filts']
teacher = Res152Unet(num_classes=num_classes, in_channels=in_channels, depth=depth, start_filts=start_filts)

# Optimizer
lr = Unet_cfg['teacher']['lr']
optimizer = Adam(teacher.parameters(), lr=lr)

# Scheduler
T_0 = Unet_cfg['teacher']['T_0']
T_mult = Unet_cfg['teacher']['T_mult']
eta_min = Unet_cfg['teacher']['eta_min']
# Define the schedulers
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[3])

# Loss function
criterion = loss.WeightedCELoss()

# Accelerator setup
accelerator = Accelerator(log_with="wandb")
accelerator.init_trackers(project_name="ACTIA", config={'teacher':Unet_cfg['teacher'],'id':identity})
accelerator.trackers[0].run.name = f'basicKD::teacher_id={identity}'
# Prepare model and data for accelerator
teacher, optimizer, criterion, scheduler, train_loader, val_loader = accelerator.prepare(
    teacher, optimizer, criterion, scheduler, train_loader, val_loader
)

# teacher the model
engine(teacher, train_loader, val_loader, criterion, optimizer, scheduler, accelerator,epochs=Unet_cfg['teacher']['epochs'])
accelerator.wait_for_everyone()



"""saving the model"""
if(Unet_cfg['teacher']['save']):
    name =f"{Unet_cfg['teacher']['save_dir']}/Unet/{identity}.safetensors"
    print(f"teacher model saved to : {name}")
    unwrapped_teacher = accelerator.unwrap_model(teacher)
    safetensors.torch.save_file(unwrapped_teacher.state_dict(), name)

