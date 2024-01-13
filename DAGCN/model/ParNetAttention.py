import numpy as np
import torch
from torch import nn
from torch.nn import init



class ParNetAttention(nn.Module):

    def __init__(self, in_channels=512,out_channels=512):
        super().__init__()
        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels,out_channels,kernel_size=(1,1),padding=(0,0)),
            nn.Sigmoid()
        )

        self.conv1x1=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        self.silu=nn.SiLU()
        

    def forward(self, x):
        b, c, _, _ = x.size()

        x1=self.conv1x1(x)
        #x2=self.conv3x3(x)
        x=self.sse(x)*x1
        y=self.silu(x)
        return y
class SpatialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, dilation=1):
        super(SpatialConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding='same',
            stride=stride,
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MultiScale_GraphConvNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 dilations=[1, 2, 3, 4,5],
                 residual=True,
                 residual_kernel_size=1):
        super().__init__()

        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)

        self.branches = nn.ModuleList([
             nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                SpatialConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    pna = ParNetAttention(channel=512)
    output=pna(input)
    print(output.shape)

    