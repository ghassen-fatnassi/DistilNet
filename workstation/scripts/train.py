import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from torch.optim import Adam,SGD,lr_scheduler,AdamW,Adamax
from accelerate import Accelerator
from accelerate.utils import LoggerType

from tqdm import tqdm

from segmentationDataset import SegDataset
from  utils import datasetSplitter
import loss
from models.Unet import segUnet
from config_loader import load_yaml

"""loading config file"""
cfg=load_yaml()

"""setting up the device"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_step(model ,dataloader ,criterion ,optimizer ,accelerator):

    model.train()
    loss=0.0

    for batch,images,masks in enumerate(tqdm(dataloader)):

        images=images.to(accelerator.device)
        masks=masks.to(accelerator.device)
        
        optimizer.zero_grad()

        with accelerator.auto_cast():
            outputs=model(images)
            curr_loss=criterion(outputs, masks)
        
        accelerator.backward(loss)
        optimizer.step()

        loss+=curr_loss.item()

def val_step(model,dataloader,criterion,accelerator):
    model.eval()
    loss=0.0
    
    for batch,images,masks in enumerate(tqdm(dataloader)):

        images=images.to(accelerator.device)
        masks=masks.to(accelerator.device)

        with accelerator.auto_cast():
            with torch.inference_mode():
                outputs=model(images)
                curr_loss=criterion(outputs,masks)

        loss+=curr_loss.item()

def train_loop(model,train_loader,val_loader,criterion,optimizer,accelerator,epochs=cfg['hyperparameters']['epochs']):
    for epoch in tqdm(range(epochs)):
        return
        


train_val_data=SegDataset(mode='train')
train_loader,val_loader=datasetSplitter(train_val_data,cfg['hyperparameters']['batch_size'])

teacher=segUnet(num_classes=cfg['dataset']['num_classes'], in_channels=3, depth=2)
optimizer=AdamW(teacher.parameters(),lr=cfg['hyperparameters']['lr'])
scheduler=lr_scheduler()
criterion=loss.basicCELoss()
accelerator = Accelerator()
teacher.to(accelerator.device)
criterion.to(accelerator.device)
