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
from actia.workstation.src.Unet import Unet
from student_engine import engine

torch.manual_seed(50)


# Load configurations
cfg = utils.load_yaml()
Unet_cfg = utils.load_yaml(cfg['paths']['cfg']['Unet'])

# Set environment variables
os.environ['WANDB_API_KEY'] = cfg['wandb']['API_KEY']
os.environ["WANDB_SILENT"] = cfg['wandb']['silent']
os.environ["WANDB_DIR"] = f"{Unet_cfg['student']['log_dir']}/Unet"
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

in_channels = Unet_cfg['in_channels']
# Student Model configuration
num_classes = cfg['dataset']['num_classes']
depth = Unet_cfg['student']['depth']
start_filts = Unet_cfg['student']['start_filts']
student = Unet(num_classes=num_classes, in_channels=in_channels, depth=depth, start_filts=start_filts,negative_slope=0.01)

# Teacher Model configuration
depth = Unet_cfg['teacher']['depth']
start_filts = Unet_cfg['teacher']['start_filts']
teacher= Unet(num_classes=num_classes, in_channels=in_channels, depth=depth, start_filts=start_filts,negative_slope=0.01)
teacher.load_state_dict(safetensors.torch.load_file(Unet_cfg['student']['teacher_weight_dir']))

# Optimizer
lr = Unet_cfg['student']['lr']
optimizer = Adam(student.parameters(), lr=lr)

# Schedulers
T_0 = Unet_cfg['student']['T_0']
T_mult = Unet_cfg['student']['T_mult']
eta_min = Unet_cfg['student']['eta_min']
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=10)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[4])

# Loss function
criterion = loss.WeightedDistillationLoss(Unet_cfg['student']['temperature'],Unet_cfg['student']['epochs'],Unet_cfg['student']['alpha'])

# teacher* the model
# Accelerator setup
accelerator = Accelerator(log_with="wandb")
accelerator.init_trackers(project_name="ACTIA", config={'student':Unet_cfg['student'],'teacher':Unet_cfg['teacher'],'id':identity})
accelerator.trackers[0].run.name = f'method=basicKD_student_id={identity}'
# Prepare model and data for accelerator
student, Teacher, optimizer, criterion, scheduler, train_loader, val_loader = accelerator.prepare(
    student, teacher, optimizer, criterion, scheduler, train_loader, val_loader
)
    
engine(student,teacher, train_loader, val_loader, criterion, optimizer, scheduler, accelerator,epochs=Unet_cfg['student']['epochs'])
accelerator.wait_for_everyone()

"""saving the model"""
if(Unet_cfg['student']['save']):
    name = f"{Unet_cfg['student']['save_dir']}/Unet/{identity}.safetensors"
    print(f"student model saved to : {name}")
    unwrapped_student = accelerator.unwrap_model(student)
    safetensors.torch.save_file(unwrapped_student.state_dict(), name)
