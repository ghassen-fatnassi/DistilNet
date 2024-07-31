import torch
import torch.nn as nn
from utils import load_yaml
import numpy as np
torch.manual_seed(50)

cfg = load_yaml()

class WeightedCELoss(nn.Module):
    """Cross-entropy loss without any augmentation."""
    def __init__(self):
        super(WeightedCELoss, self).__init__()
        self.weights=torch.tensor(cfg['dataset']['class_weights'])
        
    def to(self,device):
        self.weights=self.weights.to(device)
        return super(WeightedCELoss,self).to(device) 
    #calling the to method of parent class to ensure proper handling of the rest of the logic
    
    def forward(self, out_mask, true_mask, temperature=1):
        out_mask = out_mask.float() / temperature
        true_mask = true_mask.float()

        out_mask = out_mask.permute(0, 2, 3, 1).reshape(-1, out_mask.size(1))
        true_mask = true_mask.permute(0, 2, 3, 1).reshape(-1, true_mask.size(1))

        loss = nn.functional.cross_entropy(out_mask, true_mask,weight=self.weights, reduction='mean')
        return loss

class TverskyCEDiceWeightedLoss(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass

class WeightedDistillationLoss(nn.Module):
    """Distillation loss."""
    
    def __init__(self,temperature,epochs,alpha=0.5):
        super().__init__()
        self.ce_loss = WeightedCELoss()
        self.temperature = temperature
        self.alpha = alpha
        self.epochs=epochs
        self.coeff=self.epochs/100

    def forward(self, student_out,true_mask, teacher_out):

        soft_loss = self.ce_loss(student_out, teacher_out, self.temperature) * (self.temperature ** 2)
        hard_loss = self.ce_loss(student_out, true_mask)

        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return loss
    
    def to(self,device):
        self.ce_loss=self.ce_loss.to(device)
        return super(WeightedDistillationLoss,self).to(device) 
    
    def step_alpha(self):
        self.alpha = np.exp(-self.coeff * self.alpha / self.epochs)
