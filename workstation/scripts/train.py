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
import wandb

"""loading config file"""
cfg=load_yaml()

"""setting up the device"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def train_step(model ,dataloader ,criterion ,optimizer ,accelerator):

    model.train()
    loss=0.0
    metrics={'loss':0.0,'iou':0.0,'dice':0.0,'f1':0.0,'accuracy':0.0,'precision':0.0,'recall':0.0}
    for batch,(images,masks) in enumerate(tqdm(dataloader)):

        images=images.to(accelerator.device)
        masks=masks.to(accelerator.device)
        
        optimizer.zero_grad()

        with accelerator.auto_cast():
            outputs=model(images)
            curr_loss=criterion(outputs, masks)
        
        accelerator.backward(loss)
        optimizer.step()

    loss+=curr_loss.item() 
    metrics['loss']=loss/len(dataloader)
    return metrics

def val_step(model,dataloader,criterion,accelerator):
    model.eval()
    loss=0.0
    metrics={'loss':0.0,'iou':0.0,'dice':0.0,'f1':0.0,'accuracy':0.0,'precision':0.0,'recall':0.0}
    for batch,(images,masks) in enumerate(tqdm(dataloader)):

        images=images.to(accelerator.device)
        masks=masks.to(accelerator.device)

        with accelerator.auto_cast():
            with torch.inference_mode():
                outputs=model(images)
                curr_loss=criterion(outputs,masks)
        
    loss+=curr_loss.item()
    metrics['loss']=loss/len(dataloader)
    return metrics

def engine(model,train_loader,val_loader,criterion,optimizer,scheduler,accelerator,epochs=cfg['hyperparameters']['epochs']):
    for epoch in tqdm(range(epochs)):
        epoch_train_metrics=train_step(model,train_loader,criterion,optimizer,accelerator)
        epoch_val_metrics=val_step(model,val_loader,criterion,accelerator)
        scheduler.step()
        epoch_metrics={'train':epoch_train_metrics,'val':epoch_val_metrics}
        accelerator.log(epoch_metrics,step=epoch)
    accelerator.end_training()




train_val_data=SegDataset(mode='train')
train_loader,val_loader=datasetSplitter(train_val_data,cfg['hyperparameters']['batch_size'])

teacher=segUnet(num_classes=cfg['dataset']['num_classes'], in_channels=3, depth=2, start_filts=16)
optimizer=AdamW(teacher.parameters(),lr=cfg['hyperparameters']['lr'])
scheduler=lr_scheduler().StepLR(optimizer,step_size=cfg['hyperparameters']['step_size'],gamma=cfg['hyperparameters']['gamma'])
criterion=loss.basicCELoss()
accelerator = Accelerator(log_with=["wandb", LoggerType.TENSORBOARD])

accelerator.init_trackers(
    project_name="my_project",
    config=cfg['hyperparameters']
    )

accelerator.device = device
teacher.to(accelerator.device)
criterion.to(accelerator.device)
