import torch
import torch.nn as nn
from layers import Downsample, Upsample

class Unet(nn.Module):
    def __init__(self, nside, inly, outly):
        super(Unet, self).__init__()

        self.down_stack = nn.ModuleList([
            Downsample(inly,4, 3),  # (batch_size, 1024, 1024, 8)
            Downsample(4,4, 3),  # (batch_size, 512, 512, 8)
            Downsample(4,8, 3),  # (batch_size, 256, 256, 8)
            Downsample(8,8, 3), 
            Downsample(8,8, 3),  # (batch_size, 64, 64, 8)
            Downsample(8,8, 3),  # (batch_size, 32, 32, 8)
            Downsample(8,16, 3),  # (batch_size, 16, 16, 8)
            Downsample(16,16, 3),  # (batch_size, 8, 8, 8)
            Downsample(16,16, 3),  # (batch_size, 4, 4, 8)
        ])

        self.up_stack = nn.ModuleList([
            Upsample(16,16, 4, apply_dropout=True,apply_batchnorm=False),  # (batch_size, 2, 2, 16)
            Upsample(32,16, 4, apply_dropout=True,apply_batchnorm=False),  # (batch_size, 4, 4, 8)
            Upsample(32,8, 4, apply_dropout=True,apply_batchnorm=False),  # (batch_size, 8, 8, 8)
            Upsample(16,8, 4),  # (batch_size, 16, 16, 8)
            Upsample(16,8, 4),  # (batch_size, 32, 32, 8)
            Upsample(16,8, 4),  # (batch_size, 64, 64, 8)
            Upsample(16,4, 4),  # (batch_size, 128, 128, 8)
            Upsample(8,4, 4),  # (batch_size, 256, 4, 8)
            Upsample(8,4, 4),  # (batch_size, 512, 512, 16)
        ])

        self.last = nn.ConvTranspose2d(in_channels=8, out_channels=outly, kernel_size=4, stride=2, padding=1, output_padding=0, bias=True)

    def forward(self, x):
        # Downsampling through the model
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)

        skips = list(reversed(skips[:-1]))

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            #print(f"x shape: {x.shape}, skip shape: {skip.shape}")
            x = torch.cat([x, skip], dim=1)

        x = self.last(x)

        return x


import torch
import torch.nn as nn
from layers import Downsample

class Discriminator(nn.Module):
    def __init__(self, nside, inly, outly):
        super(Discriminator, self).__init__()

        self.down1 = Downsample(inly+outly,4, 4, apply_batchnorm=False)  # (batch_size, 512, 512, 8)
        self.down2 = Downsample(4,8, 4)  # (batch_size, 256, 256, 8)
        self.down3 = Downsample(8,16, 4)  # (batch_size, 128, 128, 16)
        self.down4 = Downsample(16,32, 4)  # (batch_size, 64, 64, 16)

        self.zero_pad1 = nn.ZeroPad2d(1)  # Padding to ensure the dimensions match
        self.conv = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.zero_pad2 = nn.ZeroPad2d(1)  # Padding to ensure the dimensions match
        self.last = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=0)

    def forward(self, inp, tar):
        x = torch.cat([inp, tar], dim=1)  # (batch_size, 1024, 1024, channels*2)

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        x = self.zero_pad1(x)
        x = self.conv(x)
        x = self.batchnorm1(x)
        x = self.leaky_relu(x)

        x = self.zero_pad2(x)
        x = self.last(x)

        return x

