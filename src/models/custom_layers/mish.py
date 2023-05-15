# from RODAN model.py
import torch.nn as nn
from torch import tanh

class Mish(nn.Module):
    """
    Mish: A self regularized non-monotonic activation function
    https://arxiv.org/abs/1908.08681
    """
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x *( tanh(nn.functional.softplus(x)))