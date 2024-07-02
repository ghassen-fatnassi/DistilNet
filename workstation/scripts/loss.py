import torch
import torch.nn as nn

class basicCELoss(nn.Module):
    """cross_entropy_loss directly without any augmentation"""
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,out_mask,true_mask):
        return self.criterion(out_mask,true_mask)

    