# from RODAN model.py
import torch.nn as nn

class Swish(nn.Module):
    # TODO:  add description 
    
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())