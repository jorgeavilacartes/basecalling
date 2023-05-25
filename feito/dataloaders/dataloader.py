import numpy as np
import h5py
import torch

from pathlib import Path
from typing import Union
from torch.utils.data import Dataset

class DatasetONT(Dataset):
    "Load a single sample for training/evaluating the network"
    # TODO: remove output_network_len, it will be computed in the Trainer class
    def __init__(self, recfile: Union[str,Path], output_network_len: int):
        """
        Args:
            recfile (Union[str,Path]): path to hdf5 file with events and labels
            output_network_len (int): length of the output signal of the networkssss (required for CTC loss)
        """        
        self.recfile = recfile # hdf5 file with training data
        self.output_network_len = output_network_len

        # load metadata recfile
        h5 = h5py.File(self.recfile, "r")
        n_signals, signal_len = h5["events"].shape
        _, label_len = h5["labels"].shape
        self.metadata = dict(
            n_signal   = n_signals,  # number of signals in the file (called 'events' in the files from RODAN)
            signal_len = signal_len, # length of the signal (or event length in RODAN)
            label_len  = label_len,   # length of the labels; each position correspond to one character in {-,A,C,G,T}
            output_network_len = output_network_len
        )
        h5.close() 

    def __getitem__(self, index: int): 
        "return (signal, label) for one sample/individual in the dataset"

        # open file with signals and labels
        h5 = h5py.File(self.recfile, "r")
        signal, label = h5["events"][index], h5["labels"][index]
        h5.close()

        # include channel to signal [Channel, Signal], and load data as Tensor
        signals = torch.from_numpy(np.expand_dims(signal,axis=0))
        labels = torch.from_numpy(label)
        # print("labels from data loader", type(labels), labels)

        # define target and input lengths of the loss function
        target_lens = torch.from_numpy( np.array(len(np.trim_zeros(label))) ) # with numpy #TODO: check how to do this with pytorch
        
        # target_lens = torch.from_numpy(np.array(self.metadata["label_len"])) 
        input_lens = torch.from_numpy(np.array(self.metadata["output_network_len"]))
       
        return signals, labels, input_lens, target_lens
        
    def __len__(self,):
        "number of pairs (signal, label) in the input file"
        return self.metadata["n_signal"]