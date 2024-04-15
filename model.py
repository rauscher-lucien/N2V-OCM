import torch
import torch.nn as nn

from utils import *


class ConvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)
    

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.convblock = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_a =  self.convblock(x)
        x_b = self.pool(x_a)
        return x_a, x_b


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2, padding=0)
        self.convblock = ConvBlock(2*in_ch, out_ch)

    def forward(self, x, x_new):
        x = self.upsample(x)
        x = torch.cat([x, x_new], dim=1)
        x = self.convblock(x)
        return x



class NewUNet(nn.Module):
    def __init__(self):
        super(NewUNet, self).__init__()

        self.base = 32

        # Define the network layers using the base size
        self.down0 = DownBlock(in_ch=1, out_ch=self.base)
        self.down1 = DownBlock(in_ch=self.base, out_ch=2*self.base)
        self.down2 = DownBlock(in_ch=2*self.base, out_ch=4*self.base)
        self.down3 = DownBlock(in_ch=4*self.base, out_ch=8*self.base)
        self.conv = ConvBlock(in_ch=8*self.base, out_ch=8*self.base)
        self.up4 = UpBlock(in_ch=8*self.base, out_ch=4*self.base)
        self.up3 = UpBlock(in_ch=4*self.base, out_ch=2*self.base)
        self.up2 = UpBlock(in_ch=2*self.base, out_ch=self.base)
        self.up1 = UpBlock(in_ch=self.base, out_ch=self.base)
        self.outconv = nn.Conv2d(self.base, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # Forward pass through the UNet model
        x0_a, x0_b = self.down0(x)
        x1_a, x1_b = self.down1(x0_b)
        x2_a, x2_b = self.down2(x1_b)
        x3_a, x3_b = self.down3(x2_b)
        x4 = self.conv(x3_b)
        x3 = self.up4(x4, x3_a)
        x2 = self.up3(x3, x2_a)
        x1 = self.up2(x2, x1_a)
        x0 = self.up1(x1, x0_a)
        x = self.outconv(x0)
        return x
