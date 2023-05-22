# https://github.com/biodlab/RODAN/blob/029f7d5eb31b11b53537f13164bfedee0c0786e4/model.py#L187

import sys
import torch.nn as nn 
from .custom_layers import (
    Mish, 
    Swish, 
    ConvBlockRodan
)

from collections import namedtuple

OUTPUT_LEN=420

# data to create RODAN's architecture 
rna_default = [[-1, 256, 0, 3, 1, 1, 0], [-1, 256, 1, 10, 1, 1, 1], [-1, 256, 1, 10, 10, 1, 1], [-1, 320, 1, 10, 1, 1, 1], [-1, 384, 1, 15, 1, 1, 1], [-1, 448, 1, 20, 1, 1, 1], [-1, 512, 1, 25, 1, 1, 1], [-1, 512, 1, 30, 1, 1, 1], [-1, 512, 1, 35, 1, 1, 1], [-1, 512, 1, 40, 1, 1, 1], [-1, 512, 1, 45, 1, 1, 1], [-1, 512, 1, 50, 1, 1, 1], [-1, 768, 1, 55, 1, 1, 1], [-1, 768, 1, 60, 1, 1, 1], [-1, 768, 1, 65, 1, 1, 1], [-1, 768, 1, 70, 1, 1, 1], [-1, 768, 1, 75, 1, 1, 1], [-1, 768, 1, 80, 1, 1, 1], [-1, 768, 1, 85, 1, 1, 1], [-1, 768, 1, 90, 1, 1, 1], [-1, 768, 1, 95, 1, 1, 1], [-1, 768, 1, 100, 1, 1, 1]]
# dna_default = [[-1, 320, 0, 3, 1, 1, 0], [-1, 320, 1, 3, 3, 1, 1], [-1, 384, 1, 6, 1, 1, 1], [-1, 448, 1, 9, 1, 1, 1], [-1, 512, 1, 12, 1, 1, 1], [-1, 576, 1, 15, 1, 1, 1], [-1, 640, 1, 18, 1, 1, 1], [-1, 704, 1, 21, 1, 1, 1], [-1, 768, 1, 24, 1, 1, 1], [-1, 832, 1, 27, 1, 1, 1], [-1, 896, 1, 30, 1, 1, 1], [-1, 960, 1, 33, 1, 1, 1]]

DEFAULTCONFIG = dict(
    vocab=["<PAD>", "A", "C", "G", "T"],
    activation="mish", # options: mish, swish, relu, gelu
    sqex_activation="mish", # options: mish, swish, relu, gelu
    dropout=0.1,
    sqex_reduction=32
)

Config=namedtuple("CONFIG",["vocab", "activation_layer", "sqex_activation", "dropout", "sqex_reduction"])

def activation_function(activation):
    "auxiliar function used in RODAN https://github.com/biodlab/RODAN/blob/master/model.py"
    if activation == "mish":
        return Mish
    elif activation == "swish":
        return Swish
    elif activation == "relu":
        return nn.ReLU
    elif activation == "gelu":
        return nn.GELU
    else:
        print("Unknown activation type:", activation)
        sys.exit(1)

class Rodan(nn.Module):
    def __init__(self, config=None, arch=rna_default, seqlen=4096, output_len=OUTPUT_LEN):
        super(Rodan, self).__init__()

        if config is None:
            config = Config(*DEFAULTCONFIG)
        self.seqlen = seqlen
        self.vocab = config.vocab        
        self.bn = nn.BatchNorm1d
        self.output_len = output_len
        
        activation = activation_function(config.activation.lower())
        sqex_activation = activation_function(config.sqex_activation.lower())

        self.convlayers = nn.Sequential()
        in_channels = 1
        convsize = self.seqlen

        for i, layer in enumerate(arch):
            paddingarg = layer[0]
            out_channels = layer[1]
            seperable = layer[2] 
            kernel = layer[3]
            stride = layer[4]
            sqex = layer[5]
            dodropout = layer[6]
            expansion = True

            if dodropout: dropout = config.dropout
            else: dropout = 0
            if sqex: squeeze = config.sqex_reduction
            else: squeeze = 0

            if paddingarg == -1:
                padding = kernel // 2
            else: padding = paddingarg
            if i == 0: expansion = False

            convsize = (convsize + (padding*2) - (kernel-stride))//stride

            self.convlayers.add_module(
                "conv"+str(i), 
                ConvBlockRodan(in_channels, 
                out_channels, 
                kernel, 
                stride=stride, 
                padding=padding, 
                seperable=seperable, 
                activation=activation, 
                expansion=expansion, 
                dropout=dropout, 
                squeeze=squeeze, 
                sqex_activation=sqex_activation, 
                residual=True)
                )

            in_channels = out_channels
            self.final_size = out_channels
         
        self.final = nn.Linear(self.final_size, len(self.vocab))

    def forward(self, x):
        x = self.convlayers(x)
        x = x.permute(0,2,1)
        x = self.final(x)
        x = nn.functional.log_softmax(x, 2)
        return x.permute(1, 0, 2)
    
