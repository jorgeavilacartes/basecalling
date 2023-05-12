# from RODAN model.py
import torch.nn as nn
from torch import tanh

class Mish(nn.Module):
    # TODO: add description
    def __init__(self):
        super(Mish).__init__()

    def forward(self, x):
        return x *( tanh(nn.functional.softplus(x)))