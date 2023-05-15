# from RODAN model.py
import torch.nn as nn
from torch import tanh

class Mish(nn.Module):
    # TODO: add description and link to paper
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x *( tanh(nn.functional.softplus(x)))