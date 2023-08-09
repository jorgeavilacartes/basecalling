"""
Inputs are raw reads in fast5 format
"""
import logging
logging.basicConfig(level=logging.INFO,
                    format='[FEITO-basecalling] - %(asctime)s. %(message)s',
                    datefmt='%Y-%m-%d@%H:%M:%S')

import numpy as np
import h5py
import torch
import pandas as pd 

from typing import Union, List
from pathlib import Path
from collections import namedtuple
from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import (
    load_signal, 
    split_raw_signal, 
    # preprocessing
)
from ont_fast5_api.fast5_interface import get_fast5_file

# ----
#  FIXME: Is there a better way to import this?
# To create index
import sys
sys.path.append(".")
from feito.callbacks import CSVLogger  
# ----

import logging
logger=logging.getLogger()
logger.setLevel("INFO")

_Path = Union[str,Path]
_PathFast5 = Union[_Path,List[_Path]]

VARS_INDEX=["index","path_fast5", "read_id", "subsignal_id","start", "end"]
Index = namedtuple("Index", VARS_INDEX)

class DatasetBasecalling(Dataset):
    "Load a single sample for basecalling"
    def __init__(self, path_fast5: _PathFast5, 
                 len_subsignals: int = 4096, left_trim: int= 0, right_trim: int = 0,
                 len_overlap: int = 0, preprocess_signal: bool = True, 
                 path_save_index: str = "output/basecalling/index_basecalling.csv"
                 ):
        """
        Args:
            path_fast5 (Union[str,Path]): path to fast5 file(s) with raw reads
            # output_network_len (int): length of the output signal of the networkssss (required for spliting the raw read)
        """        
        self.path_fast5 = path_fast5 if type(path_fast5)==list else [path_fast5] # hdf5 file with training data
        self.path_fast5 = self.path_fast5
        # self.output_network_len = output_network_len
        self.len_subsignals = len_subsignals
        self.kwargs_split_raw_signal = {
            "len_subsignals": len_subsignals, 
            "left_trim": left_trim, 
            "right_trim": right_trim, 
            "preprocess_signal": preprocess_signal, 
            "len_overlap": len_overlap
            }
        
        self.kwargs_index = {
            "len_subsignals": len_subsignals, 
            "left_trim": left_trim, 
            "right_trim": right_trim, 
            "preprocess_signal": False, 
            "len_overlap": len_overlap
            }

        # load reads and split them
        self.n_reads = len(self.path_fast5)

        # create index (read, portion)
        self.index_logger = CSVLogger(
            VARS_INDEX,
            out_file=path_save_index, 
            overwrite=True
            )
        
        self.create_index()
        self.index = self.index_logger.values

    def __getitem__(self, index: int): 
        "return (signal,) for one sample/individual in the dataset"

        item = self.index[index]
        return self.load_subsignal(item.path_fast5, item.start, item.end)

    def __len__(self,):
        "number signals in the index"
        return len(self.index)

    def create_index(self,):
        """Indexing of the portions of raw signals
        No preprocessing needed
        """
        index=0
        for path_fast5 in tqdm(self.path_fast5, desc="Creatind Index for reads", total=self.n_reads):
            try:
                raw_signal, read_id, len_signal = load_signal(path_fast5, scale=False) # just positions are required here
                if not len(raw_signal): 
                    print(path_fast5, read_id)
                split_signal = split_raw_signal(raw_signal, **self.kwargs_index) 

                starts = range(0, len_signal, self.len_subsignals)
                
                for subsignal_id in range(split_signal.shape[0]):
                    start = starts[subsignal_id]
                    end = start + self.len_subsignals - 1 
                    # index.append(Index(idx, fast5, read_id, subsignal_id, start, end)) 
                    # monitor values
                    self.index_logger()
                    index += 1
            except:
                continue


    def load_subsignal(self, path_fast5, start, end,):
        logging.info(f"{Path(path_fast5).stem}")
        with get_fast5_file(path_fast5, mode="r") as f5:
            for read in f5.get_reads():
                raw_signal = read.get_raw_data(scale=True)
                split_signal = split_raw_signal(raw_signal,**self.kwargs_split_raw_signal) # apply (1) trim, (2) preprocessing and (3) padding if needed
        logging.info(f"{raw_signal}")
        logging.info(f"shape: {raw_signal.shape}")
        # return the required subsignal (row in the split_signal array)
        idx_subsignal = start // self.len_subsignals
        subsignal = split_signal[ idx_subsignal ,:]
        logging.info(f"chunks is {type(split_signal)} of shape {split_signal.shape}")
        return torch.from_numpy(np.expand_dims(subsignal, axis=0)).float()