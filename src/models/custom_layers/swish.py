# from RODAN model.py
import torch.nn as nn

class Swish(nn.Module):
    """
    Searching for activation functions
    https://arxiv.org/abs/1710.05941v2
    """
    # TODO:  add description and link to paper
    
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())