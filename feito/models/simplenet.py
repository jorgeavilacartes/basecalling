import torch.nn as nn 

# output length is needed for CTC loss
OUTPUT_LEN=501

class SimpleNet(nn.Module):
    """A Conv1D net inspired in LeNet"""

    def __init__(self, n_channels = 1,  output_len=501): #n_classes = 271,):
        # call parent constructor
        super(SimpleNet, self).__init__()

        self.n_channels = n_channels
        self.output_len = output_len

        # first block
        self.conv1 = nn.Conv1d(in_channels  = n_channels, out_channels = 20, kernel_size  = 20, stride = 2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size = 10, stride = 2)

        # second block
        self.conv2 = nn.Conv1d(in_channels  = 20, out_channels = 50, kernel_size = 5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size = 10, stride = 2)
        
        # linear layer
        self.fc1 = nn.Linear(in_features = 50, out_features = 5) # this output 1021 will be affected by the chose of kernel_size
        self.relu3 = nn.ReLU()


    def forward(self, x):

        # first block: convolution
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # second block: convolution
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # permute dimension of conv output to apply linear layer to channel dimension
        x = x.permute(0,2,1)

        # third block: linear
        x = self.fc1(x) # x has shape [batch size, length, channels] (or in pytorch notation: [N, T, C])
        
        # apply log_softmax along the channel axis (alphabet)
        # (required for CTC loss, check paper https://www.cs.toronto.edu/~graves/icml_2006.pdf)
        x = nn.functional.log_softmax(input=x, dim=2)

        # output a tensor of shape [T,N,C]
        # T: lenght of output sequence
        # N: batch size
        # C: number of classes
        # [N,T,C] -> [T,N,C]    
        return x.permute(1,0,2)
