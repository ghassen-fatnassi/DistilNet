import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, SGD, AdamW, Adamax
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from accelerate import Accelerator
import safetensors.torch
import wandb

import os
import json

from ... import dataset,utils,loss
from dataset import SegDataset
from utils import datasetSplitter, load_yaml
from .. import models
from models.Unet import segUnet
from student_engine import engine

# Load configurations
cfg = load_yaml()
Unet_cfg = load_yaml(cfg['paths']['cfg']['Unet'])

# Set environment variables
os.environ['WANDB_API_KEY'] = cfg['wandb']['API_KEY']
os.environ["WANDB_SILENT"] = cfg['wandb']['silent']
os.environ["WANDB_DIR"] = Unet_cfg['distillation']['log_dir']

# Dataset and DataLoaders configuration
batch_size = Unet_cfg['training']['batch_size']
data = SegDataset()
train_loader, val_loader = datasetSplitter(data, batch_size).split() # if i wanna change the split , i just gotta change random seed in here

# Student Model configuration
num_classes = cfg['dataset']['num_classes']
in_channels = Unet_cfg['distillation']['in_channels']
depth = Unet_cfg['distillation']['depth']
start_filts = Unet_cfg['distillation']['start_filts']
student = segUnet(num_classes=num_classes, in_channels=in_channels, depth=depth, start_filts=start_filts)

# Teacher Model configuration
in_channels = Unet_cfg['training']['in_channels']
depth = Unet_cfg['training']['depth']
start_filts = Unet_cfg['training']['start_filts']
teacher= segUnet(num_classes=num_classes, in_channels=in_channels, depth=depth, start_filts=start_filts)
teacher.load_state_dict(safetensors.torch.load_file(Unet_cfg['distillation']['teacher_weight_dir']))

# Optimizer
lr = Unet_cfg['distillation']['lr']
optimizer = AdamW(student.parameters(), lr=lr)

# Scheduler
factor = Unet_cfg['distillation']['factor']
patience = Unet_cfg['distillation']['patience']
scheduler = ReduceLROnPlateau(optimizer, factor=factor, patience=patience)

# Loss function
criterion = loss.WeightedDistillationLoss(Unet_cfg['distillation']['temperature'], Unet_cfg['distillation']['alpha'])

# Training the model
if __name__ == '__main__':
    # Accelerator setup
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(project_name="ACTIA", config={'student':Unet_cfg['distillation'],'teacher':Unet_cfg['training']})

    # Prepare model and data for accelerator
    student, Teacher, optimizer, criterion, scheduler, train_loader, val_loader = accelerator.prepare(
        student, teacher, optimizer, criterion, scheduler, train_loader, val_loader
    )
        
    engine(student,teacher, train_loader, val_loader, criterion, optimizer, scheduler, accelerator,epochs=Unet_cfg['training']['epochs'],img_sampling_index=9)
    accelerator.wait_for_everyone()

    """saving the model"""
    if(Unet_cfg['distillation']['save']):
        student_depth=Unet_cfg['distillation']['depth']
        student_in_channels=Unet_cfg['distillation']['in_channels']
        student_start_filts=Unet_cfg['distillation']['start_filts']
        student_batch_size=Unet_cfg['distillation']['batch_size']
        student_epochs=Unet_cfg['distillation']['epochs']
        student_lr=Unet_cfg['distillation']['lr']

        teacher_depth=Unet_cfg['training']['depth']
        teacher_in_channels=Unet_cfg['training']['in_channels']
        teacher_start_filts=Unet_cfg['training']['start_filts']
        teacher_batch_size=Unet_cfg['training']['batch_size']
        teacher_epochs=Unet_cfg['training']['epochs']
        teacher_lr=Unet_cfg['training']['lr']

        name=f"{Unet_cfg['training']['save_dir']}/:::STUDENT:::depth{student_depth}_in{student_in_channels}_start{student_start_filts}_batch{student_batch_size}_epochs{student_epochs}_lr{student_lr}:::TEACHER:::depth{student_depth}_in{teacher_in_channels}_start{teacher_start_filts}_batch{teacher_batch_size}_epochs{teacher_epochs}_lr{teacher_lr}.safetensors"
        with open(name, "w") as f:
            pass
        unwrapped_student = accelerator.unwrap_model(student)
        safetensors.torch.save_file(unwrapped_student.state_dict(), name)
