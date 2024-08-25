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
from Unet import studentUnet,teacherUnet,Unet
from myTransformers import teacherSegformer
from response_distillation import engine

torch.manual_seed(50)


cfg = utils.load_yaml()
Unet_cfg = utils.load_yaml(cfg['paths']['cfg']['Unet'])

os.environ['WANDB_API_KEY'] = cfg['wandb']['API_KEY']
os.environ["WANDB_SILENT"] = cfg['wandb']['silent']
os.environ["WANDB_DIR"] = f"{Unet_cfg['student']['log_dir']}/Unet"
wandb.util.logger.setLevel(logging.ERROR)


def generate_timestamp_id():
    now = datetime.datetime.now()
    timestamp_id = now.strftime("%Y%m%d_%H%M%S")
    return timestamp_id
identity="2"

batch_size = Unet_cfg['teacher']['batch_size']
data = dataset.SegDataset()
train_loader, val_loader = utils.datasetSplitter(data, batch_size).split() # if i wanna change the split , i just gotta change random seed in here

in_channels = Unet_cfg['in_channels']
num_classes = cfg['dataset']['num_classes']
depth = Unet_cfg['student']['depth']
start_filts = Unet_cfg['student']['start_filts']
student = studentUnet(num_classes=num_classes, in_channels=in_channels, depth=depth, start_filts=start_filts)



depth = Unet_cfg['teacher']['depth']
start_filts = Unet_cfg['teacher']['start_filts']
teacher= teacherUnet(num_classes=num_classes, in_channels=in_channels, depth=depth, start_filts=start_filts)
teacher.load_state_dict(safetensors.torch.load_file(Unet_cfg['student']['teacher_weight_dir']))

lr = Unet_cfg['student']['lr']
optimizer = Adam(student.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)


##Coarse
criterion1 = loss.StudentLoss(epochs=Unet_cfg['student']['epochs'],temperature=Unet_cfg['student']['temperature'],alpha=Unet_cfg['student']['alpha'],method1=Unet_cfg['student']['method1'],method2=Unet_cfg['student']['method2'])
criterion2 = loss.StudentLoss(epochs=Unet_cfg['student']['epochs'],temperature=Unet_cfg['student']['temperature'],alpha=0.0,method1=Unet_cfg['student']['method1'],method2=Unet_cfg['student']['method2']) # can also be JaccardLoss , DiceLoss, FocalLoss
accelerator = Accelerator(log_with="wandb")
accelerator.init_trackers(project_name="ACTIA", config={'student':Unet_cfg['student'],'teacher':Unet_cfg['teacher'],'id':identity})
accelerator.trackers[0].run.name = f'method=basicKD_student_id={identity}'
student, teacher, optimizer, criterion1,criterion2, scheduler, train_loader, val_loader = accelerator.prepare(
    student, teacher, optimizer, criterion1,criterion2, scheduler, train_loader, val_loader
)
engine(student,teacher, train_loader, val_loader, criterion1,criterion2, optimizer, scheduler, accelerator,epochs=Unet_cfg['student']['epochs'])
accelerator.end_training()
accelerator.wait_for_everyone()

#config_change=
cfg['train_imgs_dir']='./actia/workstation/data/X/images'
cfg['train_masks_dir']='./actia/workstation/data/X/labels'
utils.change_yaml(cfg)
##Fine
lr = 0.005
optimizer = Adam(student.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)

data = dataset.SegDataset()
train_loader, val_loader = utils.datasetSplitter(data, batch_size).split() # if i wanna change the split , i just gotta change random seed in here


criterion1 = loss.StudentLoss(epochs=Unet_cfg['student']['epochs'],temperature=Unet_cfg['student']['temperature'],alpha=0.5,method1=Unet_cfg['student']['method1'],method2=Unet_cfg['student']['method2'])
criterion2 = loss.StudentLoss(epochs=Unet_cfg['student']['epochs'],temperature=Unet_cfg['student']['temperature'],alpha=0.0,method1=Unet_cfg['student']['method1'],method2=Unet_cfg['student']['method2']) # can also be JaccardLoss , DiceLoss, FocalLoss
accelerator = Accelerator(log_with="wandb")
accelerator.init_trackers(project_name="ACTIA", config={'student':Unet_cfg['student'],'teacher':Unet_cfg['teacher'],'id':identity})
accelerator.trackers[0].run.name = f'method=basicKD_student_id={identity}'
student, teacher, optimizer, criterion1,criterion2, scheduler, train_loader, val_loader = accelerator.prepare(
    student, teacher, optimizer, criterion1,criterion2, scheduler, train_loader, val_loader
)
engine(student,teacher, train_loader, val_loader, criterion1,criterion2, optimizer, scheduler, accelerator,epochs=Unet_cfg['student']['epochs'])
accelerator.end_training()
accelerator.wait_for_everyone()

"""saving the model"""
if(Unet_cfg['student']['save']):
    name = f"{Unet_cfg['student']['save_dir']}/Unet/student_{identity}.safetensors"
    print(f"student model saved to : {name}")
    unwrapped_student = accelerator.unwrap_model(student)
    safetensors.torch.save_file(unwrapped_student.state_dict(), name)
