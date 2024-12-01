import torch
import torch.nn as nn

class Downsample(nn.Module):
    def __init__(self, in_channels,out_channels, size, apply_batchnorm=True):
        super(Downsample, self).__init__()
        initializer = nn.init.normal_

        self.apply_batchnorm = apply_batchnorm

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=size, stride=1, padding='same', bias=False)
        initializer(self.conv1.weight, 0.0, 0.02)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=out_channels, kernel_size=size, stride=1, padding='same', bias=False)
        initializer(self.conv2.weight, 0.0, 0.02)

        if self.apply_batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_channels)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)

        if self.apply_batchnorm:
            x = self.batchnorm(x)

        x = self.leaky_relu(x)
        return x

import torch
import torch.nn as nn

class Upsample(nn.Module):
    def __init__(self, in_channels,out_channels ,size, apply_dropout=False,apply_batchnorm=True):
        super(Upsample, self).__init__()
        initializer = nn.init.normal_

        self.conv_transpose = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=size, stride=2, padding=1, output_padding=0, bias=False)
        initializer(self.conv_transpose.weight, 0.0, 0.02)
        self.apply_batchnorm=apply_batchnorm
        if self.apply_batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_channels)

        self.apply_dropout = apply_dropout
        if self.apply_dropout:
            self.dropout = nn.Dropout(0.2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x)

        if self.apply_dropout:
            x = self.dropout(x)

        x = self.relu(x)
        return x

