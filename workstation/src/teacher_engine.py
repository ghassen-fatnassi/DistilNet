from tqdm.notebook import trange,tqdm
import numpy as np
import torch
import wandb
from segmentation_models_pytorch.metrics import iou_score, f1_score, recall, get_stats

from utils import  load_yaml,UnetMaskProcessor

torch.manual_seed(50)

# Loading configuration file
cfg = load_yaml()
Unet_cfg = load_yaml(cfg['paths']['cfg']['Unet'])

def ready_kernels(model):

    encoders=model.encoders
    bottleneck=model.bottleneck
    decoders=model.decoders
    kernels=[]

    for module in encoders:
        kernels.append(module.conv.conv1.weight.data.clone()[-3,-1])
        kernels.append(module.conv.conv2.weight.data.clone()[-3,-1])
        kernels.append(module.conv.conv3.weight.data.clone()[-3,-1])

    kernels.append(bottleneck.conv1.weight.data.clone()[-1,-1])
    kernels.append(bottleneck.conv2.weight.data.clone()[-1,-1])
    kernels.append(bottleneck.conv3.weight.data.clone()[-1,-1])

    for module in decoders:
        kernels.append(module.conv.conv1.weight.data.clone()[-3,-1])
        kernels.append(module.conv.conv2.weight.data.clone()[-3,-1])
        kernels.append(module.conv.conv3.weight.data.clone()[-3,-1])

    wandb_kernels=[]
    for kernel in kernels:
        wandb_kernels.append(wandb.Image(kernel.permute(0,1).cpu().numpy()))
    return wandb_kernels

def return_loggable_imgs(images, masks):

    if isinstance(images, torch.Tensor):
        images = images.permute(0, 2, 3, 1).cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.permute(0, 2, 3, 1).cpu().numpy()
    
    single_channel_masks = np.argmax(masks, axis=-1)
    class_labels = UnetMaskProcessor().class_labels
    wandb_images = []

    for img, mask in zip(images, single_channel_masks):
        wandb_images.append(wandb.Image(img, masks={"predictions": {"mask_data": mask, "class_labels": class_labels}}))
    
    return wandb_images

def ready_internal_masks(internal_masks):
    if isinstance(internal_masks, torch.Tensor):
        internal_masks = internal_masks.cpu().numpy()
    wandb_masks=[]
    for mask in internal_masks:
        wandb_masks.append(wandb.Image(mask))
    return wandb_masks

def return_batch_metrics(criterion, outputs, masks):

    metrics = {'loss': 0.0, 'miou': 0.0, 'f1': 0.0, 'recall': 0.0}
    tp, fp, fn, tn = get_stats(outputs, masks, mode='multilabel', threshold=0.5)
    metrics['loss'] = criterion(outputs, masks)
    metrics['miou'] = iou_score(tp, fp, fn, tn, reduction="micro")
    metrics['f1'] = f1_score(tp, fp, fn, tn, reduction="micro")
    metrics['recall'] = recall(tp, fp, fn, tn, reduction="micro-imagewise")
    return metrics

def train_step(model, dataloader, criterion, optimizer, accelerator, epoch):

    if Unet_cfg['last_batch']==-1:
        last_batch = len(dataloader)
    else:
        last_batch= Unet_cfg['last_batch']
    model.train()
    metrics = {'loss': 0.0, 'miou': 0.0, 'f1': 0.0, 'recall': 0.0, 'epoch':epoch}
    tracked_images = None
    tracked_masks = None
    internal_masks=None
    for batch, (images, masks) in enumerate(tqdm(dataloader)):
        with accelerator.autocast():
            outputs = model(images)
            batch_metrics = return_batch_metrics(criterion, outputs, masks)
        
        optimizer.zero_grad()
        accelerator.backward(batch_metrics['loss'])
        optimizer.step()
        
        batch_metrics['loss'] = batch_metrics['loss'].item()
        accelerator.log({"train_batches": batch_metrics, "batch": last_batch * epoch + batch})

        if batch == Unet_cfg['img_sampling_index'] & epoch%Unet_cfg['log_masks_every']==0:
            tracked_images = images
            tracked_masks = outputs
            internal_masks=model.exp_forward(images)
        if batch >= last_batch:
            break

    for key in metrics.keys():
        if key!='epoch':
            metrics[key] += batch_metrics[key] / last_batch


    return metrics, tracked_images, tracked_masks, internal_masks

def val_step(model, dataloader, criterion, accelerator, epoch):
    if Unet_cfg['last_batch']==-1:
        last_batch = len(dataloader)
    else:
        last_batch= Unet_cfg['last_batch']
    model.eval()
    metrics = {'loss': 0.0, 'miou': 0.0, 'f1': 0.0, 'recall': 0.0,'epoch':epoch}
    tracked_images = None
    tracked_masks = None
    internal_masks=None

    for batch, (images, masks) in enumerate(tqdm(dataloader)):
        with accelerator.autocast():
            with torch.inference_mode():
                outputs = model(images)
                batch_metrics = return_batch_metrics(criterion, outputs, masks)
        
        batch_metrics['loss'] = batch_metrics['loss'].item()
        accelerator.log({"val_batches": batch_metrics, "batch": last_batch * epoch + batch})


        if batch == Unet_cfg['img_sampling_index'] & epoch%Unet_cfg['log_masks_every']==0:
            tracked_images = images
            tracked_masks = outputs
            internal_masks=model.exp_forward(images)
        if batch >= last_batch:
            break

    for key in metrics.keys():
        if key!='epoch':
            metrics[key] += batch_metrics[key] / last_batch


    return metrics, tracked_images, tracked_masks,internal_masks

def engine(model, train_loader, val_loader, criterion, optimizer, scheduler, accelerator, epochs):

    for epoch in trange(epochs):
        epoch_train_metrics = train_step(model, train_loader, criterion, optimizer, accelerator, epoch)
        epoch_val_metrics = val_step(model, val_loader, criterion, accelerator, epoch)
        scheduler.step(epoch_train_metrics[0]['loss'])
        if epoch%Unet_cfg['log_masks_every']==0:
            val_tracked_images = epoch_val_metrics[1]
            val_tracked_masks = epoch_val_metrics[2]
            val_wandb_img_and_masks = return_loggable_imgs(val_tracked_images, val_tracked_masks)
            val_wandb_internal_representations=ready_internal_masks(epoch_val_metrics[3])
            train_tracked_images = epoch_val_metrics[1]
            train_tracked_masks = epoch_val_metrics[2]
            train_wandb_img_and_masks = return_loggable_imgs(train_tracked_images, train_tracked_masks)
            train_wandb_internal_representations=ready_internal_masks(epoch_train_metrics[3])
            learned_kernels=ready_kernels(model)
            epoch_metrics = {
                'train_epochs': epoch_train_metrics[0],
                'val_epochs': epoch_val_metrics[0],
                'epoch': epoch,
                'val_mask_per_epoch': val_wandb_img_and_masks,
                'val_internals_per_epoch':val_wandb_internal_representations,
                'filters_per_epoch': learned_kernels,
                'train_mask_per_epoch': train_wandb_img_and_masks,
                'train_internals_per_epoch':train_wandb_internal_representations,
            }
        else:
            epoch_metrics = {
                'train_epochs': epoch_train_metrics[0],
                'val_epochs': epoch_val_metrics[0],
                'epoch': epoch,
            }
        accelerator.log(epoch_metrics)
    
    accelerator.end_training()
