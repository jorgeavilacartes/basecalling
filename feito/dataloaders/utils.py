"""Helper functions for basecalling"""

import numpy as np
from typing import Optional
from ont_fast5_api.fast5_interface import get_fast5_file

def load_signal(path_fast5: str):
    "Load a signal from fast5 file"    
    with get_fast5_file(path_fast5, mode="r") as f5:
        for read in f5.get_reads():
            raw_signal = read.get_raw_data()
            len_signal = len(raw_signal)
            read_id    = read.read_id
            
    return raw_signal, read_id, len_signal


def split_raw_signal(signal: np.ndarray, len_subsignals: int = 4096, left_trim: int= 0, right_trim: int = 0, len_overlap: int = 0, preprocess_signal: bool = True):
    """Return an array with non-overlapping signals of the same length.
    First the signal is trimmed (left/right), then the signal is padded with 0 
    in the end to have perfect subsignals of len_subsignal's lengths

    Args:
        signal (np.ndarray): input raw read
        len_subsignals (int, optional): fixed length of signals, it must be the input of the basecaller. Defaults to 4096.
        len_overlap (int): _description_. Defaults to 0.
        left_trim (int, optional): _description_. Defaults to 0.
        right_trim (int, optional): _description_. Defaults to 0.
    """    
    len_signal = len(signal)

    # trim signal
    start = left_trim - 1 if left_trim > 0 else 0 
    end   = len_signal - right_trim + 1 if right_trim > 0 else len_signal
    trimmed_signal = signal[start:end].copy()

    if preprocess_signal:
        trimmed_signal = preprocessing(trimmed_signal)

    # pad signal at the end with zeros to make the length divisible by len_subsignals
    len_padd = len_subsignals - (len(trimmed_signal) % len_subsignals)
    trimmed_signal = np.pad(trimmed_signal, (0,len_padd), 'constant', constant_values=(0,0))
    
    # reshape trimmed signal
    return trimmed_signal.reshape((-1,len_subsignals))


def preprocessing(signal, factor=1.4826):
    """
    Apply preprocessing to the entire raw signal.
    Same as in RODAN
    https://github.com/biodlab/RODAN/blob/029f7d5eb31b11b53537f13164bfedee0c0786e4/basecall.py#L80C13-L81C67
    """
    med = np.median(signal)
    mad = np.median(np.absolute(signal - med)) * factor
    return (signal - med) / mad