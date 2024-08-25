from tqdm.rich import trange, tqdm
import numpy as np
import torch
import wandb
from segmentation_models_pytorch.metrics import iou_score, recall, get_stats
from utils import load_yaml, UnetMaskProcessor

torch.manual_seed(50)

cfg = load_yaml()
Unet_cfg = load_yaml(cfg['paths']['cfg']['Unet'])

def return_loggable_imgs(images, masks):
    if isinstance(images, torch.Tensor):
        images = images.permute(0, 2, 3, 1).cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.permute(0, 2, 3, 1).cpu().numpy()

    single_channel_masks = np.argmax(masks, axis=-1)
    class_labels = UnetMaskProcessor().class_labels
    wandb_images = []

    for img, pred_mask in zip(images, single_channel_masks):
        wandb_images.append(wandb.Image(img, masks={"predictions": {"mask_data": pred_mask, "class_labels": class_labels}}))
    return wandb_images

def return_batch_metrics(criterion, outputs, masks):
    metrics = {'loss': 0.0, 'miou': 0.0, 'recall': 0.0}
    tp, fp, fn, tn = get_stats(outputs.argmax(dim=1), masks.argmax(dim=1), mode='multiclass', num_classes=19)
    metrics['loss'] = criterion(outputs, masks)
    metrics['miou'] = iou_score(tp, fp, fn, tn, reduction="micro")
    metrics['recall'] = recall(tp, fp, fn, tn, reduction="micro")
    return metrics

def train_step(model, dataloader, criterion, optimizer, accelerator, epoch):
    last_batch = len(dataloader) if Unet_cfg['last_batch'] == -1 else Unet_cfg['last_batch']
    model.train()
    metrics = {'loss': 0.0, 'miou': 0.0, 'recall': 0.0}
    tracked_images, tracked_masks, internal_masks = None, None, None

    for batch, (images, masks) in enumerate(dataloader):
        with accelerator.autocast():
            outputs = model(images)
            batch_metrics = return_batch_metrics(criterion, outputs, masks)

        accelerator.backward(batch_metrics['loss'] / Unet_cfg['teacher']['grad_acc'])
        if (batch + 1) % Unet_cfg['teacher']['grad_acc'] == 0:
            optimizer.step()
            optimizer.zero_grad()

        batch_metrics['loss'] = batch_metrics['loss'].item()
        accelerator.log({"train_batches": batch_metrics, "batch": last_batch * epoch + batch})

        if batch == Unet_cfg['img_sampling_index'] & epoch % Unet_cfg['log_masks_every'] == 0:
            tracked_images, tracked_masks = images, outputs

        if batch >= last_batch:
            break

        for key in metrics.keys():
            metrics[key] += batch_metrics[key] / last_batch

    return metrics, tracked_images, tracked_masks, internal_masks

def val_step(model, dataloader, criterion, accelerator, epoch):
    last_batch = len(dataloader) if Unet_cfg['last_batch'] == -1 else Unet_cfg['last_batch']
    model.eval()
    metrics = {'loss': 0.0, 'miou': 0.0, 'recall': 0.0}
    tracked_images, tracked_masks = None, None

    for batch, (images, masks) in enumerate(dataloader):
        with accelerator.autocast():
            with torch.inference_mode():
                outputs = model(images)
                batch_metrics = return_batch_metrics(criterion, outputs, masks)

        batch_metrics['loss'] = batch_metrics['loss'].item()
        accelerator.log({"val_batches": batch_metrics, "batch": last_batch * epoch + batch})

        if batch == Unet_cfg['img_sampling_index'] & epoch % Unet_cfg['log_masks_every'] == 0:
            tracked_images, tracked_masks = images, outputs

        if batch >= last_batch:
            break

        for key in metrics.keys():
            metrics[key] += batch_metrics[key] / last_batch

    return metrics, tracked_images, tracked_masks

def engine(model, train_loader, val_loader, criterion, optimizer, scheduler, accelerator, epochs):
    for epoch in trange(epochs):
        epoch_train_metrics = train_step(model, train_loader, criterion, optimizer, accelerator, epoch)
        epoch_val_metrics = val_step(model, val_loader, criterion, accelerator, epoch)
        scheduler.step()
        if epoch % Unet_cfg['log_masks_every'] == 0:
            val_tracked_images = epoch_val_metrics[1]
            val_tracked_masks = epoch_val_metrics[2]
            val_wandb_img_and_masks = return_loggable_imgs(val_tracked_images, val_tracked_masks)
            epoch_metrics = {
                'train_epochs': epoch_train_metrics[0],
                'val_epochs': epoch_val_metrics[0],
                'epoch': epoch,
                'val_mask_per_epoch': val_wandb_img_and_masks,
                'lr': scheduler.get_last_lr()[0]
            }
        else:
            epoch_metrics = {
                'train_epochs': epoch_train_metrics[0],
                'val_epochs': epoch_val_metrics[0],
                'epoch': epoch,
                'lr': scheduler.get_last_lr()[0]
            }
        accelerator.log(epoch_metrics)
    
