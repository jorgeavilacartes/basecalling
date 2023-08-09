"""
This class assumes that the input are fast5 files with ONE entire raw read
Input is a single fast5 file or a directory with many fast5 files.
"""

import logging
logging.basicConfig(level=logging.INFO,
                    format='[FEITO-basecalling] - %(asctime)s. %(message)s',
                    datefmt='%Y-%m-%d@%H:%M:%S')

# builtin libraries
import re
from typing import Union, Optional
from pathlib import Path
from collections import namedtuple, defaultdict
from functools import partial

# secondary libraries
import numpy as np
# import parasail
import torch
import torch.nn as nn
from torch.utils.data import DataLoader # load batches to the network
from fast_ctc_decode import beam_search, viterbi_search
from tqdm import tqdm

# types
_Path = Union[Path,str]


class Basecaller:

    def __init__(self, model, device, basecalling_loader, 
                 path_fasta: Optional[_Path] = None, rna: bool = True, use_viterbi = True,
                 return_reads: bool = False
                 ):
        self.model  = model.to(device) # model with pretrained weigths loaded
        self.device = device
        self.basecalling_loader = basecalling_loader # load signals
        self.batch_size  = basecalling_loader.batch_size
        self.path_fasta  = path_fasta # to save basecalled raw-reads (if not None)
        self.rna = rna 
        self.alphabet    = "NACGU" if rna else "NACGT"
        print(f"Alphabet Basecaller: {self.alphabet} | rna: {rna}")
        self.use_viterbi = use_viterbi
        self.search_algo = viterbi_search if use_viterbi else beam_search
        self.return_reads = return_reads 

        if Path(self.path_fasta).is_file():
            Path(self.path_fasta).unlink() # remove file

        Path(self.path_fasta).parent.mkdir(exist_ok=True, parents=True)

    def __call__(self,):
        
        "Returns a list with accuracies and another list with basecalled signals"
        basecalled_signals = []
        n_batches=len(self.basecalling_loader)
        idx = 0 
        with tqdm(total=n_batches, leave=True, ncols=100, bar_format='{l_bar}{bar}| [{elapsed}{postfix}]') as progress_bar:

            for n_batch, batch in enumerate(self.basecalling_loader):
                X = batch.to(self.device)
                if n_batch == 0: 
                    print(X.shape)
                    print(X)
                progress_bar.set_description(f"Evaluating | Batch: {n_batch+1}/{n_batches}")
                basecalled_signals_batch = self.basecall_one_batch(X)
                
                if self.return_reads:
                    basecalled_signals.extend(basecalled_signals_batch)
                
                # progress_bar.set_postfix(train_loss='%.4f' % current_avg_loss)

                # send basecalled reads to a fasta file
                progress_bar.update(1)
                with open(self.path_fasta, "a") as fp: 
                    for read in basecalled_signals_batch:
                        fp.write(f">{idx}\n")
                        fp.write(read+"\n")
                        idx +=1
        
        if self.return_reads: 
            return basecalled_signals
        else:
            return None

    @torch.no_grad()
    def basecall_one_batch(self, X):
        "Return basecalled signals in the chosen alphabet"
        preds  = self.model(X) # preds shape: (len-signal, item, size-alphabet)

        # torch.softmax(torch.tensor(pre[:,i,:]), dim=-1)

        if self.device == "cpu":
            basecalled_signals = list(
                self.signal_to_read(
                    signal=torch.softmax(preds[:,item,:], dim=-1).detach().numpy(), use_viterbi=self.use_viterbi
                ) 
                for item in range(preds.shape[1])
                )
        else:
            basecalled_signals = list(
                self.signal_to_read(
                    signal=torch.softmax(preds[:,item,:], dim=-1).cpu().detach().numpy(), use_viterbi=self.use_viterbi
                ) 
                for item in range(preds.shape[1])
                )
        return basecalled_signals

    def signal_to_read(self, signal, use_viterbi: bool = True):
        "Apply viterbi or beam search to a signal"
        
        if use_viterbi is True:
            seq, path = viterbi_search(signal, self.alphabet) 
        else:
            seq, path = beam_search(signal, self.alphabet, beam_size=5, beam_cut_threshold=0.1)

        return seq