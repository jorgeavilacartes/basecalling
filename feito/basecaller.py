"""
This class assumes that for each signal we have the ground truth,
so no transcriptome is needed here. It must be used before hand 
to create the test dataset (same format than training and validation sets).

Input datasets are in hdf5 format
"""
# builtin libraries
import re
import logging
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
                 path_fasta: Optional[_Path] = None, rna: bool = True, use_viterbi = True):
        self.model  = model.to(device) # model with pretrained weigths loaded
        self.device = device
        self.basecalling_loader = basecalling_loader # load signals
        self.batch_size  = basecalling_loader.batch_size
        self.path_fasta  = path_fasta # to save basecalled raw-reads (if not None)
        self.rna = rna 
        self.alphabet    = "NACGU" if rna else "NACGT"
        self.use_viterbi = use_viterbi
        self.search_algo = viterbi_search if use_viterbi else beam_search

        Path(self.path_fasta).parent.mkdir(exist_ok=True, parents=True)

    def __call__(self,):
        
        "Returns a list with accuracies and another list with basecalled signals"
        basecalled_signals = []
        n_batches=len(self.basecalling_loader)
        idx = 0 
        with tqdm(total=n_batches, leave=True, ncols=100, bar_format='{l_bar}{bar}| [{elapsed}{postfix}]') as progress_bar:

            for n_batch, batch in enumerate(self.basecalling_loader):

                progress_bar.set_description(f"Evaluating | Batch: {n_batch+1}/{n_batches}")
                basecalled_signals_batch = self.basecall_one_batch(batch)
                
                basecalled_signals.extend(basecalled_signals_batch)
                
                # progress_bar.set_postfix(train_loss='%.4f' % current_avg_loss)

                # send basecalled reads to a fasta file
                progress_bar.update(1)
                with open(self.path_fasta, "a") as fp: 
                    for read in basecalled_signals_batch:
                        fp.write(f">{idx}\n")
                        fp.write(read+"\n")
                        idx +=1

        return basecalled_signals

    def basecall_one_batch(self, X):
        "Return basecalled signals in the chosen alphabet"
        preds  = self.model(X) # preds shape: (len-signal, item, size-alphabet)
        basecalled_signals = list(
            self.signal_to_read(signal=preds[:,item,:].detach().numpy(), use_viterbi=self.use_viterbi, rna=self.rna) 
            for item in range(preds.shape[1])
            )

        return basecalled_signals

    def label_to_alphabet(self, label):
        """Map vector of integers to sequence in DNA or RNA alphabet
        blanks are not considered.
        """
        
        return "".join([self.int2char[i] for i in label if i > 0])

    def signal_to_read(self, signal, use_viterbi: bool = True, rna: bool = True):
        "Apply viterbi or beam search to a signal"
        
        if use_viterbi is True:
            seq, path = viterbi_search(signal, self.alphabet) 
        else:
            seq, path = beam_search(signal, self.alphabet, beam_size=5, beam_cut_threshold=0.1)

        return seq