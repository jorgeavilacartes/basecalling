"""
Inputs are raw reads in fast5 format
"""
import numpy as np
import h5py
import torch
import pandas as pd 

from typing import Union, List
from pathlib import Path
from collections import namedtuple
from torch.utils.data import Dataset
from .utils import load_signal, split_raw_signal, preprocessing
from ont_fast5_api.fast5_interface import get_fast5_file
import logging
logger=logging.getLogger()
logger.setLevel("INFO")

_Path = Union[str,Path]
_PathFast5 = Union[_Path,List[_Path]]

Index = namedtuple("Index",["index","path_fast5", "read_id", "subsignal_id","start", "end"])

class DatasetBasecalling(Dataset):
    "Load a single sample for basecalling"
    def __init__(self, path_fast5: _PathFast5, 
                 len_subsignals: int = 4096, left_trim: int= 0, right_trim: int = 0,
                 len_overlap: int = 0, preprocess_signal: bool = True, path_save_index: str = "output/index_basecalling.csv"):
        """
        Args:
            path_fast5 (Union[str,Path]): path to fast5 file(s) with raw reads
            # output_network_len (int): length of the output signal of the networkssss (required for spliting the raw read)
        """        
        self.path_fast5 = path_fast5 if type(path_fast5)==list else [path_fast5] # hdf5 file with training data
        # self.output_network_len = output_network_len
        self.len_subsignals = len_subsignals
        self.kwargs = {"len_subsignals": len_subsignals, "left_trim": left_trim, "right_trim": right_trim, "preprocess_signal": preprocess_signal, "len_overlap": len_overlap}

        # load reads and split them
        self.n_reads = len(self.path_fast5)

        # create index (read, portion)
        self.index = self.create_index()

        # check if path_fasta exists
        assert not Path(path_save_index).is_file(), f"path_save_index already exists: {path_save_index}"
        Path(path_save_index).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.index).to_csv(path_save_index, sep="\t")


    def __getitem__(self, index: int): 
        "return (signal,) for one sample/individual in the dataset"

        item = self.index[index]
        return self.load_subsignal(item.path_fast5, item.start, item.end)

    def __len__(self,):
        "number of pairs (signal, label) in the input file"
        return len(self.index)

    def create_index(self,):
        "Indexing of the portions of raw signals"
        idx=0
        index = []
        for fast5 in self.path_fast5:
            try:
                raw_signal, read_id, len_signal = load_signal(fast5)
                if not len(raw_signal): 
                    print(fast5, read_id)
                split_signal = split_raw_signal(raw_signal, **self.kwargs) # apply (1) trim, (2) preprocessing and (3) padding if needed

                starts = range(0, len_signal, self.len_subsignals)
                
                # logger.info(f"read info: {fast5} | {read_id}")
                # logger.info(f"split signal columns {split_signal.shape[0]}")

                for idx_subsignal in range(split_signal.shape[0]):
                    start = starts[idx_subsignal]
                    end = start + self.len_subsignals - 1 
                    index.append(Index(idx, fast5, read_id, idx_subsignal, start, end)) 
                    idx += 1
            except:
                continue
        return index

    def load_subsignal(self, path_fast5, start, end):
        with get_fast5_file(path_fast5, mode="r") as f5:
            for read in f5.get_reads():
                raw_signal = read.get_raw_data()
                split_signal = split_raw_signal(raw_signal)
        subsignal = split_signal[start // self.len_subsignals,:]
        return torch.from_numpy(np.expand_dims(subsignal, axis=0)).float()
