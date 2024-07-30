import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(50)

class conv_block(nn.Module):
    def __init__(self, in_c, out_c,negative_slope):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1,bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, inputs):
        return self.relu(self.bn(self.conv3(self.relu(self.conv2(self.relu(self.conv1(inputs)))))))

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c,negative_slope):
        super().__init__()
        self.conv = conv_block(in_c, out_c, negative_slope)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c,negative_slope):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = conv_block(2 * out_c, out_c,negative_slope)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class segUnet(nn.Module):
    def __init__(self, num_classes, in_channels=3, depth=5, start_filts=64,negative_slope=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.negative_slope=negative_slope
        self.internal_masks=[]
        self.skips=[]

        """ Encoders """
        self.encoders = nn.ModuleList([encoder_block(in_channels, start_filts,self.negative_slope)])
        self.encoders.extend([encoder_block(start_filts * (2 ** i), start_filts * (2 ** (i + 1)),self.negative_slope*(i*3)) for i in range(depth - 1)])

        """ Bottleneck """
        self.bottleneck = conv_block(start_filts * (2 ** (depth - 1)), start_filts * (2 ** depth),self.negative_slope)

        """ Decoders """
        self.decoders = nn.ModuleList([decoder_block(start_filts * (2 ** i), start_filts * (2 ** (i - 1)),self.negative_slope*(i*3)) for i in range(depth, 0, -1)])

        """ Classifier """
        self.outputs = nn.Conv2d(start_filts, num_classes, kernel_size=1)

    def forward(self, inputs):
        x = inputs
        self.skips=[]
        for encoder in self.encoders:
            x, p = encoder(x)
            self.skips.append(x)
            x = p

        x = self.bottleneck(x)

        for i, decoder in enumerate(self.decoders):
            x = decoder(x, self.skips[-(i+1)])
        outputs = self.outputs(x)
        return outputs
    
    def exp_forward(self,inputs):
        self.skips = []
        self.internal_masks=[]
        x = inputs
        self.internal_masks.append(x[-1])
        for encoder in self.encoders:
            x, p = encoder(x)
            self.internal_masks.append(x[-1,-1])
            self.skips.append(x)
            x = p
        self.internal_masks.append(x[-1,-1])
        x = self.bottleneck(x)

        for i, decoder in enumerate(self.decoders):
            self.internal_masks.append(x[-1,-1])
            x = decoder(x, self.skips[-(i+1)])
        
        return self.internal_masks
        