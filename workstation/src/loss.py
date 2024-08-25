import torch
import torch.nn as nn
# from utils import load_yaml
import numpy as np
torch.manual_seed(50)
import segmentation_models_pytorch as smp

class HardLoss(nn.Module):
    """multi_method_loss"""
    def __init__(self,**kwargs):
        super(HardLoss, self).__init__()
        self.__dict__.update(**kwargs)
        self.loss_fn=self.loss_factory()
        
    def forward(self, out_mask, true_mask):
        return self.loss_fn(out_mask,torch.argmax(true_mask,dim=1))

    def loss_factory(self):
        if self.method == "CELoss":
            return nn.CrossEntropyLoss(reduction='mean')
        elif self.method == "JaccardLoss":
            return smp.losses.JaccardLoss("multiclass",from_logits=True)
        elif self.method == "FocalLoss":
            return smp.losses.FocalLoss("multiclass")
        elif self.method == "DiceLoss":
            return smp.losses.DiceLoss("multiclass",from_logits=True)
        else:
            raise ValueError(f"Unknown distillation method: {self.method}")

class KLDiv(nn.Module):
    """KL_div_loss"""
    def __init__(self,temperature):
        super(KLDiv, self).__init__()
        self.temperature=temperature
        self.loss_func = nn.KLDivLoss(reduction="batchmean",log_target=True)

    def forward(self, student_out, teacher_out):
        teacher_log_probs = torch.log_softmax(teacher_out / (self.temperature), dim=1)
        student_log_probs = torch.log_softmax(student_out / (self.temperature), dim=1)
        return (self.temperature**2)*self.loss_func(student_log_probs,teacher_log_probs)/(student_out.size(2)**2) #the division is very important

class OT(nn.Module):
    """optimal transport"""
    def __init__(self,**kwargs):
        super(OT, self).__init__()
    
    def forward(self, student_out,teacher_out):
        pass    

class DistillationLoss(nn.Module):

    def __init__(self,**kwargs):
        super(DistillationLoss, self).__init__()
        self.__dict__.update(kwargs)
        self.loss_fn = self.loss_factory(**kwargs)

    def forward(self, student_out,teacher_out):
        return self.loss_fn(student_out,teacher_out)
    
    def loss_factory(self,**kwargs):
        if self.method == "KL_DIV":
            return KLDiv(self.temperature)
        elif self.method == "OT":
            return OT(**kwargs)
        else:
            raise ValueError(f"Unknown distillation loss: {self.method}")
        
class StudentLoss(nn.Module):
    def __init__(self,alpha=0.8,epochs=None,**kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.hard_loss= HardLoss(method=self.method1,**kwargs)
        self.soft_loss= DistillationLoss(method=self.method2,**kwargs)
        self.epochs=epochs
        self.alpha=alpha
        if self.epochs!=None:
            self.coeff=self.epochs/100

    def forward(self, student_out,teacher_out,true_mask):

        SL = self.soft_loss(student_out, teacher_out)
        HL = self.hard_loss(student_out, true_mask)

        return self.alpha * SL + (1 - self.alpha) * HL
    
    def step_alpha(self,curr_epoch):
        if self.epochs!=None:
            self.alpha = self.alpha+(-(self.alpha-0.2)/self.epochs)*curr_epoch