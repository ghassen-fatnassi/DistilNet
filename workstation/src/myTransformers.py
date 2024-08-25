import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor,AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import v2

class teacherSegformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model=SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")

    def forward(self,inputs):
        return self.model(inputs).logits
     
    def to(self,device):
        self.model=self.model.to(device)
        return super().to(device)
    
class teacherMask2former(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-cityscapes-semantic")

    def forward(self,inputs):
        inputs=v2.Resize((1024,1024))(inputs)
        return self.model(inputs).logits
     
    def to(self,device):
        self.model=self.model.to(device)
        return super().to(device)