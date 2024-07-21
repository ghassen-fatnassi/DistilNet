from tqdm import trange, tqdm
import numpy as np
import torch
import wandb
from segmentation_models_pytorch.metrics import iou_score, f1_score, recall, get_stats

from ....utils import  load_yaml,cityscapesMaskProcessor,bdd10kMaskProcessor

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

def ready_internal_masks(internal_masks):
    if isinstance(internal_masks, torch.Tensor):
        internal_masks = internal_masks.cpu().numpy()
    wandb_masks=[]
    for mask in internal_masks:
        wandb_masks.append(wandb.Image(mask))
    return wandb_masks

def return_loggable_imgs(images, masks):

    if isinstance(images, torch.Tensor):
        images = images.permute(0, 2, 3, 1).cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.permute(0, 2, 3, 1).cpu().numpy()
    
    single_channel_masks = np.argmax(masks, axis=-1)
    class_labels = bdd10kMaskProcessor().class_labels
    wandb_images = []

    for img, mask in zip(images, single_channel_masks):
        wandb_images.append(wandb.Image(img, masks={"predictions": {"mask_data": mask, "class_labels": class_labels}}))
    
    return wandb_images

def return_batch_metrics(criterion, teacher_outputs, student_outputs, masks):
    metrics = {'loss': 0.0, 'miou': 0.0, 'f1': 0.0, 'recall': 0.0}
    tp, fp, fn, tn = get_stats(student_outputs, masks, mode='multilabel', threshold=0.5)  # threshold rounds the output to 0 or 1
    metrics['loss'] = criterion(student_outputs, masks, teacher_outputs)
    metrics['miou'] = iou_score(tp, fp, fn, tn, reduction="micro")
    metrics['f1'] = f1_score(tp, fp, fn, tn, reduction="micro")
    metrics['recall'] = recall(tp, fp, fn, tn, reduction="micro-imagewise")
    return metrics

def train_step(student, teacher, dataloader, criterion, optimizer, accelerator, epoch):
    if Unet_cfg['distillation']['last_batch']==-1:
        last_batch = len(dataloader)
    else:
        last_batch= Unet_cfg['distillation']['last_batch']
    student.train()
    teacher.eval()
    metrics = {'loss': 0.0, 'miou': 0.0, 'f1': 0.0, 'recall': 0.0, 'epoch':epoch}

    for batch, (images, masks) in enumerate(tqdm(dataloader)):
        with accelerator.autocast():
            student_outputs = student(images)
            with torch.inference_mode():
                teacher_outputs = teacher(images)
            batch_metrics = return_batch_metrics(criterion, teacher_outputs, student_outputs, masks)
        
        optimizer.zero_grad()
        accelerator.backward(batch_metrics['loss'])
        optimizer.step()

        batch_metrics['loss'] = batch_metrics['loss'].item()
        accelerator.log({"train_batches": batch_metrics, "batch": last_batch * epoch + batch+1})

        if batch >= last_batch:
            break

    for key in metrics.keys():
        if key!='epoch':
            metrics[key] += batch_metrics[key] / last_batch
        

    return metrics

def val_step(student, teacher, dataloader, criterion, accelerator, epoch, img_sampling_index):
    if Unet_cfg['distillation']['last_batch']==-1:
        last_batch = len(dataloader)
    else:
        last_batch= Unet_cfg['distillation']['last_batch']    
    student.eval()
    teacher.eval()
    metrics = {'loss': 0.0, 'miou': 0.0, 'f1': 0.0, 'recall': 0.0, 'epoch':epoch}
    tracked_images = None
    tracked_masks = None
    internal_masks = None

    for batch, (images, masks) in enumerate(tqdm(dataloader)):
        with accelerator.autocast():
            with torch.inference_mode():
                student_outputs = student(images)
                teacher_outputs = teacher(images)
                batch_metrics = return_batch_metrics(criterion, teacher_outputs, student_outputs, masks)
        
        batch_metrics['loss'] = batch_metrics['loss'].item()
        accelerator.log({"val_batches": batch_metrics,"batch":last_batch * epoch + batch+1})

        # If the batch is the one we want to sample
        if batch == img_sampling_index and epoch % Unet_cfg['distillation']['log_masks_every'] == 0:
            tracked_images = images
            tracked_masks = student_outputs
            internal_masks = student(images, visualize_mask=True)

        if batch >= last_batch:
            break

    for key in metrics.keys():
        if key!='epoch':
            metrics[key] += batch_metrics[key] / last_batch

    return metrics, tracked_images, tracked_masks, internal_masks

def engine(student, teacher, train_loader, val_loader, criterion, optimizer, scheduler, accelerator, epochs, img_sampling_index):
    for epoch in trange(epochs):
        epoch_train_metrics = train_step(student, teacher, train_loader, criterion, optimizer, accelerator, epoch)
        epoch_val_metrics = val_step(student, teacher, val_loader, criterion, accelerator, epoch, img_sampling_index)
        scheduler.step(epoch_train_metrics['loss'])
        
        if epoch % Unet_cfg['training']['log_masks_every'] == 0:
            tracked_images = epoch_val_metrics[1]
            tracked_masks = epoch_val_metrics[2]
            wandb_img_and_masks = return_loggable_imgs(tracked_images, tracked_masks)
            wandb_internal_representations = ready_internal_masks(epoch_val_metrics[3])
            learned_kernels = ready_kernels(student)
            epoch_metrics = {
                'train_epochs': epoch_train_metrics,
                'val_epochs': epoch_val_metrics[0],
                'epoch': epoch,
                'mask_per_epoch': wandb_img_and_masks,
                'internals_per_epoch': wandb_internal_representations,
                'filters_per_epoch': learned_kernels
            }
        else:
            epoch_metrics = {
                'train_epochs': epoch_train_metrics,
                'val_epochs': epoch_val_metrics[0],
                'epoch': epoch,
            }
        accelerator.log(epoch_metrics)
    
    accelerator.end_training()
