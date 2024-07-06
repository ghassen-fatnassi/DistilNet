import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.bn1(self.conv1(inputs)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = conv_block(2 * out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class segUnet(nn.Module):
    def __init__(self, num_classes, in_channels=3, depth=5, start_filts=64):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        """ Encoders """
        self.encoders = nn.ModuleList([encoder_block(in_channels, start_filts)])
        self.encoders.extend([encoder_block(start_filts * (2 ** i), start_filts * (2 ** (i + 1))) for i in range(depth - 1)])

        """ Bottleneck """
        self.bottleneck = conv_block(start_filts * (2 ** (depth - 1)), start_filts * (2 ** depth))

        """ Decoders """
        self.decoders = nn.ModuleList([decoder_block(start_filts * (2 ** i), start_filts * (2 ** (i - 1))) for i in range(depth, 0, -1)])

        """ Classifier """
        self.outputs = nn.Conv2d(start_filts, num_classes, kernel_size=1)

    def forward(self, inputs):
        skips = []
        x = inputs
        for encoder in self.encoders:
            x, p = encoder(x)
            skips.append(x)
            x = p

        x = self.bottleneck(x)

        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skips[-(i+1)])

        outputs = self.outputs(x)
        return outputs
