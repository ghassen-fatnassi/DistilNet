from tqdm import trange, tqdm
import numpy as np
import torch
import wandb
from utils import load_yaml, cityscapesMaskProcessor
from segmentation_models_pytorch.metrics import iou_score, f1_score, recall, get_stats

# Loading configuration file
cfg = load_yaml()
Unet_cfg = load_yaml(cfg['paths']['cfg']['Unet'])

def return_loggable_imgs(images, masks):
    if isinstance(images, torch.Tensor):
        images = images.permute(0, 2, 3, 1).cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.permute(0, 2, 3, 1).cpu().numpy()
    
    single_channel_masks = np.argmax(masks, axis=1)
    class_labels = cityscapesMaskProcessor().class_labels
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
    last_batch = 10
    student.train()
    teacher.eval()
    metrics = {'loss': 0.0, 'miou': 0.0, 'f1': 0.0, 'recall': 0.0}

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

        for key in metrics.keys():
            metrics[key] += batch_metrics[key] / last_batch
        
        if batch >= last_batch:
            break

    return metrics

def val_step(student, teacher, dataloader, criterion, accelerator, epoch, img_sampling_index):
    last_batch = 10
    student.eval()
    teacher.eval()
    metrics = {'loss': 0.0, 'miou': 0.0, 'f1': 0.0, 'recall': 0.0}
    tracked_images = None
    tracked_masks = None

    for batch, (images, masks) in enumerate(tqdm(dataloader)):
        with accelerator.autocast():
            with torch.inference_mode():
                student_outputs = student(images)
                teacher_outputs = teacher(images)
                batch_metrics = return_batch_metrics(criterion, teacher_outputs, student_outputs, masks)
        
        batch_metrics['loss'] = batch_metrics['loss'].item()
        accelerator.log({"val_batches": batch_metrics,"batch":last_batch * epoch + batch+1})

        for key in metrics.keys():
            metrics[key] += batch_metrics[key] / last_batch

        # If the batch is the one we want to sample
        if batch == img_sampling_index:
            tracked_images = images
            tracked_masks = masks

        if batch >= last_batch:
            break

    return metrics, tracked_images, tracked_masks

def engine(student, teacher, train_loader, val_loader, criterion, optimizer, scheduler, accelerator, epochs, img_sampling_index):
    for epoch in trange(epochs):
        epoch_train_metrics = train_step(student, teacher, train_loader, criterion, optimizer, accelerator, epoch)
        epoch_val_metrics = val_step(student, teacher, val_loader, criterion, accelerator, epoch, img_sampling_index)
        scheduler.step(epoch_train_metrics['loss'])
        
        tracked_images = epoch_val_metrics[1]
        tracked_masks = epoch_val_metrics[2]
        wandb_images = return_loggable_imgs(tracked_images, tracked_masks)
        
        epoch_metrics = {
            'train_epochs': epoch_train_metrics,
            'val_epochs': epoch_val_metrics[0],
            "epoch": epoch,
            "ex_per_epoch": wandb_images
        }
        accelerator.log(epoch_metrics)
    
    accelerator.end_training()
