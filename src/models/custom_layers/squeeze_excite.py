# from RODAN
import torch.nn as nn

class SqueezeExcite(nn.Module):
    # TODO: add description
    
    def __init__(self, in_channels = 512, size=1, reduction="/16", activation=nn.GELU):

        super(SqueezeExcite, self).__init__()
        self.in_channels = in_channels
        self.avg = nn.AdaptiveAvgPool1d(1)
        if type(reduction) == str:
            self.reductionsize = self.in_channels // int(reduction[1:])
        else:
            self.reductionsize = reduction
        self.fc1 = nn.Linear(self.in_channels, self.reductionsize)
        self.activation = activation()
        self.fc2 = nn.Linear(self.reductionsize, self.in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg(x)
        x = x.permute(0,2,1)
        x = self.activation(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return input * x.permute(0,2,1)