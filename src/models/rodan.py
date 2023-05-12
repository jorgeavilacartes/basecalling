import torch.nn as nn 
# FIXME: clean code and write missing parts
# https://github.com/biodlab/RODAN/blob/029f7d5eb31b11b53537f13164bfedee0c0786e4/model.py#L187
class Rodan(nn.Module):
    def __init__(self, config=None, arch=None, seqlen=4096, debug=False):
        super(Rodan).__init__()
        
        # if debug: print("Initializing network")
        
        self.seqlen = seqlen
        self.vocab = config.vocab        
        self.bn = nn.BatchNorm1d

        if arch == None: arch = rna_default

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
            # if debug:
            #     print("padding:", padding, "seperable:", seperable, "ch", out_channels, "k:", kernel, "s:", stride, "sqex:", sqex, "drop:", dropout, "expansion:", expansion)
            #     print("convsize:", convsize)
            self.convlayers.add_module(
                "conv"+str(i), 
                convblock(in_channels, 
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
        # if debug: print("Finished init network")

    def forward(self, x):
        #x = self.embedding(x)
        x = self.convlayers(x)
        x = x.permute(0,2,1)
        x = self.final(x)
        x = torch.nn.functional.log_softmax(x, 2)
        return x.permute(1, 0, 2)