import torch
from torch.optim import AdamW, RMSprop
from torch.optim.lr_scheduler import StepLR
from accelerate import Accelerator
import safetensors.torch
import wandb
import logging
import datetime
import os
import dataset, utils, loss
from Unet import StudentUnetWithDropout,TeacherUnetWithDropout,Unet
from myTransformers import teacherSegformer
from teacher_training import engine

torch.manual_seed(50)

cfg = utils.load_yaml()
Unet_cfg = utils.load_yaml(cfg['paths']['cfg']['Unet'])

os.environ['WANDB_API_KEY'] = cfg['wandb']['API_KEY']
os.environ["WANDB_DIR"] = f"{Unet_cfg['teacher']['log_dir']}/Unet"
wandb.util.logger.setLevel(logging.ERROR)

def generate_timestamp_id():
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")

identity = "perfecto"

num_classes = cfg['dataset']['num_classes']
in_channels = Unet_cfg['in_channels']
depth = Unet_cfg['teacher']['depth']
start_filts = Unet_cfg['teacher']['start_filts']
teacher = StudentUnetWithDropout(num_classes=num_classes, in_channels=in_channels, depth=depth, start_filts=start_filts)
for param in teacher.model.encoder.parameters():
    param.requires_grad = False
lr = Unet_cfg['teacher']['lr']
optimizer = AdamW(teacher.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=0.8)

criterion = loss.HardLoss(method=Unet_cfg['teacher']['method_fine'])

accelerator = Accelerator(log_with="wandb")
accelerator.init_trackers(project_name="ACTIA", config={'id': identity})
accelerator.trackers[0].run.name = f'res50_64_4={identity}'

batch_size = Unet_cfg['teacher']['batch_size']

data = dataset.SegDataset()
train_loader, val_loader = utils.datasetSplitter(data, batch_size).split()
teacher, optimizer, criterion, scheduler, train_loader, val_loader = accelerator.prepare(
    teacher, optimizer, criterion, scheduler, train_loader, val_loader)
engine(teacher, train_loader, val_loader, criterion, optimizer, scheduler, accelerator, epochs=Unet_cfg['teacher']['epochs'])
accelerator.end_training()
accelerator.wait_for_everyone()

if Unet_cfg['teacher']['save']:
    name = f"{Unet_cfg['teacher']['save_dir']}/Unet/baseline_teacher_{identity}.safetensors"
    print(f"teacher model saved to : {name}")
    unwrapped_teacher = accelerator.unwrap_model(teacher)
    safetensors.torch.save_file(unwrapped_teacher.state_dict(), name)
