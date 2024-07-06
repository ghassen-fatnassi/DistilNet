import torch
import torch.nn as nn
from config.config_loader import load_yaml
cfg = load_yaml()

class BasicCELoss(nn.Module):
    """Cross-entropy loss without any augmentation."""
    def __init__(self):
        super(BasicCELoss, self).__init__()
        self.weights=torch.tensor(cfg['dataset']['class_weights'])

    def forward(self, out_mask, true_mask, temperature=1):
        out_mask = out_mask.float() / temperature
        true_mask = true_mask.float()

        out_mask = out_mask.permute(0, 2, 3, 1).reshape(-1, out_mask.size(1))
        true_mask = true_mask.permute(0, 2, 3, 1).reshape(-1, true_mask.size(1))

        loss = nn.functional.cross_entropy(out_mask, true_mask.argmax(dim=1),weight=self.weights, reduction='mean')

        return loss

class BasicDistillationLoss(nn.Module):
    """Distillation loss."""
    def __init__(self,temperature,alpha=0.5):
        super(BasicDistillationLoss, self).__init__()
        self.ce_loss = BasicCELoss()

    def forward(self, true_mask, student_out, teacher_out, temperature, alpha=0.5):

        soft_loss = self.ce_loss(student_out, teacher_out, temperature) * (temperature ** 2)
        hard_loss = self.ce_loss(student_out, true_mask)

        loss = alpha * soft_loss + (1 - alpha) * hard_loss

        return loss
