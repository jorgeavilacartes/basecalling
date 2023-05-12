# A simple and lighweight neural network
import torch.nn as nn

# FIXME: check RODAN's code for the right output

class SimpleNet(nn.Module):
    """A Conv1D net inspired in LeNet"""

    def __init__(self, n_channels = 1, n_classes = 271):
        # call parent constructor
        super(SimpleNet, self).__init__()

        # first block
        self.conv1 = nn.Conv1d(
            in_channels  = n_channels, out_channels = 20, kernel_size  = 5, stride = 1
        )
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(
            kernel_size = 2, stride = 2
            )

        # second block
        self.conv2 = nn.Conv1d(
            in_channels  = 20, out_channels = 50, kernel_size = 5
            )
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(
            kernel_size = 2, stride = 2
        )
        
        # linear layer
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(
            in_features = 51050, out_features = 500
        )
        self.relu3 = nn.ReLU()

        # output linear layer
        self.fc2 = nn.Linear(
            in_features = 500, out_features = n_classes
        )
    
    def forward(self, x):

        # first block: convolution
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # second block: convolution
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # third block: linear
        x = self.flatten1(x)
        x = self.fc1(x)
        x = self.relu3(x)

        # output linear layer
        output = self.fc2(x)

        return output