import torch
import torch.nn as nn
from torch.utils.data import DataLoader # load batches to the network

from basecaller_trainer import BasecallerTrainer as Trainer
from models import SimpleNet, Rodan
from loss_functions import ctc_label_smoothing_loss
from dataloader import DatasetONT
from callbacks import CSVLogger



# network to use
model=SimpleNet()
# model=Rodan()
model_output_len = model.output_len # another way to obtain the output of the model https://github.com/biodlab/RODAN/blob/029f7d5eb31b11b53537f13164bfedee0c0786e4/model.py#L317
loss_fn = nn.CTCLoss() #ctc_label_smoothing 
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)