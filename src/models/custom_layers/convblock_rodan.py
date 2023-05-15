import torch
import torch.nn as nn
from . import SqueezeExcite

class ConvBlockRodan(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, seperable=True, expansion=True, batchnorm=True, dropout=0.1, activation=nn.GELU, sqex=True, squeeze=32, sqex_activation=nn.GELU, residual=True):
        # no bias?
        super(ConvBlockRodan, self).__init__()
        self.seperable = seperable
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.activation = activation
        self.squeeze = squeeze
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual
        self.doexpansion = expansion
        # fix self.squeeze
        dwchannels = in_channels
        if seperable:
            if self.doexpansion and self.in_channels != self.out_channels:
                self.expansion = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, groups=1, bias=False)
                self.expansion_norm = nn.BatchNorm1d(out_channels)
                self.expansion_act = self.activation()
                dwchannels = out_channels 

            self.depthwise = nn.Conv1d(dwchannels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=out_channels//groups)
            if self.batchnorm:
                self.bn1 = nn.BatchNorm1d(out_channels)
            self.act1 = self.activation()
            if self.squeeze:
                self.sqex = SqueezeExcite(in_channels=out_channels, reduction=self.squeeze, activation=sqex_activation)
            self.pointwise = nn.Conv1d(out_channels, out_channels, kernel_size=1, dilation=dilation, bias=bias, padding=0)
            if self.batchnorm:
                self.bn2 = nn.BatchNorm1d(out_channels)
            self.act2 = self.activation()
            if self.dropout:
                self.drop = nn.Dropout(self.dropout)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
            if self.batchnorm:
                self.bn1 = nn.BatchNorm1d(out_channels)
            self.act1 = self.activation()
            if self.squeeze:
                self.sqex = SqueezeExcite(in_channels=out_channels, reduction=self.squeeze, activation=sqex_activation)
            if self.dropout:
                self.drop = nn.Dropout(self.dropout)
        if self.residual and self.stride == 1:
            self.rezero = nn.Parameter(torch.Tensor([0]), requires_grad=True)

    def forward(self, x):
        orig = x

        if self.seperable:
            if self.in_channels != self.out_channels and self.doexpansion:
                x = self.expansion(x)
                x = self.expansion_norm(x)
                x = self.expansion_act(x)
            x = self.depthwise(x)
            if self.batchnorm: x = self.bn1(x)
            x = self.act1(x)
            if self.squeeze:
                x = self.sqex(x)
            x = self.pointwise(x)
            if self.batchnorm: x = self.bn2(x)
            x = self.act2(x) 
            if self.dropout: x = self.drop(x)
        else:
            x = self.conv(x)
            if self.batchnorm: x = self.bn1(x)
            x = self.act1(x)
            if self.dropout: x = self.drop(x)

        if self.residual and self.stride == 1 and self.in_channels == self.out_channels and x.shape[2] == orig.shape[2]:
            return orig + self.rezero * x # rezero
            #return orig + x # normal residual
        else:
            return x