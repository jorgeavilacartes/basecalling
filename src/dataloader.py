import h5py
from pathlib import Path
from typing import Union
from torch.utils.data import Dataset, DataLoader

class DatasetONT(Dataset):

    def __init__(self, recfile: Union[str,Path],): # seq_len: int = 4096):
        self.recfile = recfile # hdf5 file with training data

        # load metadata recfile
        h5 = h5py.File(self.recfile, "r")
        n_signals, signal_len = h5["events"].shape
        _, label_len = h5["labels"].shape
        self.metadata = dict(
            n_signal   = n_signals,  # number of signals in the file (called 'events' in the files from RODAN)
            signal_len = signal_len, # length of the signal (or event length in RODAN)
            label_len  = label_len   # length of the labels; each position correspond to one character in {A,C,G,T,-}
        )
        h5.close() 

    def __getitem__(self, index): 
        "return (signal, label) for one sample/individual in the dataset"
        # "RODAN returns (signal, signal length, label, label length)"
        # open file with signals and labels
        h5 = h5py.File(self.recfile, "r")
        signal, label = h5["events"][index], h5["labels"][index]
        h5.close()
        return signal, label

    def __len__(self,):
        "number of pairs (signal, label) in the input file"
        return self.metadata["n_signals"]