import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from torch.optim import Adam,SGD,AdamW,Adamax
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
from accelerate import Accelerator
from accelerate.utils import LoggerType

from tqdm import trange,tqdm

from segmentationDataset import SegDataset
from  utils import datasetSplitter
import loss
from models.Unet import segUnet
from config.config_loader import load_yaml
import wandb
import os


"""loading config file"""
cfg=load_yaml()

"""setting wandb api key as an environment variable && changing log directory"""
os.environ['WANDB_API_KEY'] = cfg['API_KEYS']['wandb']
os.environ["WANDB_DIR"] = os.path.abspath("/media/gaston/gaston1/DEV/ACTIA/workstation/logs/Teachers/Unet")
os.environ["WANDB_SILENT"] = "true"

"""setting up the device"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_step(model ,dataloader ,criterion ,optimizer ,accelerator,epoch):

    model.train()
    loss=0.0
    metrics={'loss':0.0}
    for batch,(images,masks) in enumerate(tqdm(dataloader)):

        images=images.to(accelerator.device)
        masks=masks.to(accelerator.device)
        
        optimizer.zero_grad()

        with accelerator.autocast():
            outputs=model(images)
            curr_loss=criterion(outputs, masks)
            loss+=curr_loss.item() 
        accelerator.log({"batch_loss":curr_loss.item()},step=len(dataloader)*epoch+batch)
        accelerator.backward(curr_loss)
        optimizer.step()

    metrics['loss']=loss/len(dataloader)
    return metrics

def val_step(model,dataloader,criterion,accelerator,epoch):
    model.eval()
    loss=0.0
    metrics={'loss':0.0}
    for batch,(images,masks) in enumerate(tqdm(dataloader)):

        images=images.to(accelerator.device)
        masks=masks.to(accelerator.device)

        with accelerator.autocast():
            with torch.inference_mode():
                outputs=model(images)
                curr_loss=criterion(outputs,masks)
                loss+=curr_loss.item()
        accelerator.log({"batch_loss":curr_loss.item()},step=len(dataloader)*epoch+batch)

    metrics['loss']=loss/len(dataloader)
    return metrics

def engine(model,train_loader,val_loader,criterion,optimizer,scheduler,accelerator,epochs=cfg['hyperparameters']['epochs']):
    for epoch in trange(epochs):
        epoch_train_metrics=train_step(model,train_loader,criterion,optimizer,accelerator,epoch)
        epoch_val_metrics=val_step(model,val_loader,criterion,accelerator,epoch)
        scheduler.step()
        epoch_metrics={'train':epoch_train_metrics,'val':epoch_val_metrics}
        accelerator.log(epoch_metrics,step=epoch)
    accelerator.end_training()




train_val_data=SegDataset(mode='train')
train_loader,val_loader=datasetSplitter(train_val_data,cfg['hyperparameters']['batch_size']).split()

teacher=segUnet(num_classes=cfg['dataset']['num_classes'], in_channels=3, depth=2, start_filts=1)
optimizer=AdamW(teacher.parameters(),lr=cfg['hyperparameters']['lr'])
scheduler=StepLR(optimizer,step_size=cfg['hyperparameters']['step_size'],gamma=cfg['hyperparameters']['gamma'])
criterion=loss.BasicCELoss()
accelerator = Accelerator(log_with="wandb")


accelerator.init_trackers(
    project_name="my_project",
    config=cfg['hyperparameters'],
    )


teacher, optimizer, criterion, scheduler, train_loader, val_loader=accelerator.prepare(teacher, optimizer, criterion, scheduler, train_loader, val_loader)

if __name__ == '__main__':

    engine(teacher,train_loader,val_loader,criterion,optimizer,scheduler,accelerator)

wandb_tracker = accelerator.get_tracker("wandb")
wandb_tracker.finish()

