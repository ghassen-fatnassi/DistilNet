from tqdm.rich import trange, tqdm
import numpy as np
import torch
import wandb
from segmentation_models_pytorch.metrics import iou_score, recall, get_stats
from utils import load_yaml, UnetMaskProcessor

torch.manual_seed(50)

cfg = load_yaml()
Unet_cfg = load_yaml(cfg['paths']['cfg']['Unet'])

def return_loggable_imgs(images, student_masks, teacher_masks):
    if isinstance(images, torch.Tensor):
        images = images.permute(0, 2, 3, 1).cpu().numpy()
    if isinstance(student_masks, torch.Tensor):
        student_masks = student_masks.permute(0, 2, 3, 1).cpu().numpy()
    if isinstance(teacher_masks, torch.Tensor):
        teacher_masks = teacher_masks.permute(0, 2, 3, 1).cpu().numpy()

    student_single_channel_masks = np.argmax(student_masks, axis=-1)
    teacher_single_channel_masks = np.argmax(teacher_masks, axis=-1)
    class_labels = UnetMaskProcessor().class_labels
    wandb_images = []

    for img, student_mask, teacher_mask in zip(images, student_single_channel_masks, teacher_single_channel_masks):
        wandb_images.append(wandb.Image(img, masks={"predictions": {"mask_data": student_mask, "class_labels": class_labels}}))
        wandb_images.append(wandb.Image(img, masks={"predictions": {"mask_data": teacher_mask, "class_labels": class_labels}}))
    
    return wandb_images

def return_batch_metrics(criterion, teacher_outputs, student_outputs, masks):
    metrics = {'loss': 0.0, 'miou': 0.0, 'recall': 0.0}
    tp, fp, fn, tn = get_stats(student_outputs.argmax(dim=1), masks.argmax(dim=1), mode='multiclass', num_classes=19)
    metrics['loss'] = criterion(student_outputs, teacher_outputs, masks)
    metrics['miou'] = iou_score(tp, fp, fn, tn, reduction="micro")
    metrics['recall'] = recall(tp, fp, fn, tn, reduction="micro")
    return metrics

def train_step(student, teacher, dataloader, criterion, optimizer, accelerator, epoch):
    last_batch = len(dataloader) if Unet_cfg['last_batch'] == -1 else Unet_cfg['last_batch']
    student.train()
    teacher.eval()
    metrics = {'loss': 0.0, 'miou': 0.0, 'recall': 0.0}
    for batch, (images, masks) in enumerate(dataloader):
        with accelerator.autocast():
            student_outputs = student(images)
            with torch.inference_mode():
                teacher_outputs = teacher(images)
            batch_metrics = return_batch_metrics(criterion, teacher_outputs, student_outputs, masks)
        
        accelerator.backward(batch_metrics['loss'] / Unet_cfg['student']['grad_acc'])
        if (batch + 1) % Unet_cfg['student']['grad_acc'] == 0:
            optimizer.step()
            optimizer.zero_grad()

        batch_metrics['loss'] = batch_metrics['loss'].item()
        accelerator.log({"train_batches": batch_metrics, "batch": last_batch * epoch + batch})
            
        if batch >= last_batch:
            break

        for key in metrics.keys():
            metrics[key] += batch_metrics[key] / last_batch
        
    return metrics

def val_step(student, teacher, dataloader, criterion, accelerator, epoch):
    last_batch = len(dataloader) if Unet_cfg['last_batch'] == -1 else Unet_cfg['last_batch']    
    student.eval()
    teacher.eval()
    metrics = {'loss': 0.0, 'miou': 0.0, 'recall': 0.0}
    tracked_images = tracked_student_masks = tracked_teacher_masks = None
    for batch, (images, masks) in enumerate(dataloader):
        with accelerator.autocast():
            with torch.inference_mode():
                student_outputs = student(images)
                teacher_outputs = teacher(images)
                batch_metrics = return_batch_metrics(criterion, teacher_outputs, student_outputs, masks)
        
        batch_metrics['loss'] = batch_metrics['loss'].item()
        accelerator.log({"val_batches": batch_metrics, "batch": last_batch * epoch + batch})

        if batch == Unet_cfg['img_sampling_index'] & epoch % Unet_cfg['log_masks_every'] == 0:
            tracked_images = images
            tracked_student_masks = student_outputs
            tracked_teacher_masks = teacher_outputs

        if batch >= last_batch:
            break

        for key in metrics.keys():
            metrics[key] += batch_metrics[key] / last_batch

    return metrics, tracked_images, tracked_student_masks, tracked_teacher_masks

def engine(student, teacher, train_loader, val_loader, criterion1, criterion2, optimizer, scheduler, accelerator, epochs):
    for epoch in trange(epochs):
        epoch_train_metrics = train_step(student, teacher, train_loader, criterion1, optimizer, accelerator, epoch)
        epoch_val_metrics = val_step(student, teacher, val_loader, criterion2, accelerator, epoch)
        scheduler.step()
        if Unet_cfg['student']['decay_alpha']:
            criterion1.step_alpha(epoch+1)  
        if epoch % Unet_cfg['log_masks_every'] == 0:
            val_tracked_images = epoch_val_metrics[1]
            val_tracked_student_masks = epoch_val_metrics[2]
            val_tracked_teacher_masks = epoch_val_metrics[3]
            
            wandb_img_and_masks = return_loggable_imgs(val_tracked_images, val_tracked_student_masks, val_tracked_teacher_masks)
            epoch_metrics = {
                'train_epochs': epoch_train_metrics,
                'val_epochs': epoch_val_metrics[0],
                'epoch': epoch,
                'mask_per_epoch': wandb_img_and_masks,
                'lr': scheduler.get_last_lr()[0],
                'alpha':criterion1.alpha
            }
        else:
            epoch_metrics = {
                'train_epochs': epoch_train_metrics,
                'val_epochs': epoch_val_metrics[0],
                'epoch': epoch,
                'lr': scheduler.get_last_lr()[0],
                'alpha': criterion1.alpha 

            }
        accelerator.log(epoch_metrics)
    
