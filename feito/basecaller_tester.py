# builtin libraries
import re
import logging
from typing import Union, Optional
from pathlib import Path
from collections import namedtuple, defaultdict
from functools import partial

# secondary libraries
import numpy as np
import parasail
import torch
import torch.nn as nn
from torch.utils.data import DataLoader # load batches to the network
from fast_ctc_decode import beam_search, viterbi_search
from tqdm import tqdm

# types
_Path = Union[Path,str]

class BasecallerTester:
    
    split_cigar = re.compile(r"(?P<len>\d+)(?P<op>\D+)")
    
    def __init__(self, model, device, test_loader, path_fasta: Optional[_Path] = None, rna: bool = True, use_viterbi = True):
        self.model  = model.to(device) # model with pretrained weigths loaded
        self.device = device
        self.test_loader = test_loader # load signals
        self.batch_size  = test_loader.batch_size
        self.path_fasta  = path_fasta # to save basecalled raw-reads (if not None)
        self.rna = rna 
        self.alphabet    = "NACGU" if rna else "NACGT"
        self.use_viterbi = use_viterbi
        self.search_algo = viterbi_search if use_viterbi else beam_search

        # set evaluation/inference mode
        self.model.eval()

        # map integers to characters in the alphabet
        self.int2char = {i:c for i,c in enumerate(self.alphabet.replace("N",""), start=1)}

    def __call__(self, return_basecalled_signals: bool=True):
        print("Que me dice")
        # inference
        accuracies, basecalled_signals = self.accuracy_all_dataset()

        accuracy = np.array(accuracies).mean()

        if self.path_fasta:
            # create parent directory if it does not exists
            Path(self.path_fasta).parent.mkdir(exist_ok=True, parents=True)

            # save basecalled signals to a fasta file
            with open(self.path_fasta, "w") as fp:
                for j,read in enumerate(basecalled_signals):
                    fp.write(f">signal_{j}\n")
                    fp.write(read + "\n")
        
        if return_basecalled_signals:
            return accuracy, basecalled_signals
        
        return accuracy
    
    def basecall_one_batch(self, X):
        "Return basecalled signals in the chosen alphabet"

        preds  = self.model(X) # preds shape: (len-signal, item, size-alphabet)
        basecalled_signals = list(
            self.signal_to_read(signal=preds[:,item,:].detach().numpy(), use_viterbi=self.use_viterbi, rna=self.rna) 
            for item in range(preds.shape[1])
            )

        return basecalled_signals

    def label_to_alphabet(self, label):
        "Map vector of integers to sequence in DNA or RNA alphabet"
        
        return "".join([self.int2char[i] for i in label if i > 0])
    
    def accuracy_one_batch(self, batch):

        X, y, output_len, target_len = (x.to(self.device) for x in batch)

        basecalled_signals = self.basecall_one_batch(X)
        ground_truth = np.apply_along_axis(lambda l: self.label_to_alphabet(l), 1, y.detach().numpy()) 
        accuracy_batch = [self.accuracy(ref=gt, seq=bs) for gt,bs in zip(basecalled_signals, ground_truth)]
        
        return accuracy_batch, basecalled_signals
    
    def accuracy_all_dataset(self,):
        "Returns a list with accuracies and another list with basecalled signals"
        basecalled_signals = []
        accuracies = []
        n_batches=len(self.test_loader)
    
        with tqdm(total=n_batches, leave=True, ncols=100, bar_format='{l_bar}{bar}| [{elapsed}{postfix}]') as progress_bar:

            for n_batch, batch in enumerate(self.test_loader):

                progress_bar.set_description(f"Evaluating | Batch: {n_batch+1}/{n_batches}")
                accuracy_batch, basecalled_signals_batch = self.accuracy_one_batch(batch)
                
                accuracies.extend(accuracy_batch)
                basecalled_signals.extend(basecalled_signals_batch)
                
                # progress_bar.set_postfix(train_loss='%.4f' % current_avg_loss)
                progress_bar.update(1)

        return accuracies, basecalled_signals

    def signal_to_read(self, signal, use_viterbi: bool = True, rna: bool = True):
        "Apply viterbi or beam search to a signal"
        
        if use_viterbi is True:
            seq, path = viterbi_search(signal, self.alphabet) 
        else:
            seq, path = beam_search(signal, self.alphabet, beam_size=5, beam_cut_threshold=0.1)

        return seq

    def accuracy(self, ref, seq, balanced=False, min_coverage=0.0):
        # From https://github.com/nanoporetech/bonito/blob/655feea4bca17feb77957c7f8be5077502292bcf/bonito/util.py#L354
        """
        Calculate the accuracy between `ref` and `seq`
        """
        # alignment = parasail.sw_trace_striped_32(seq, ref, 8, 4, parasail.dnafull) # this crashed, no meaningful error message
        alignment = parasail.sw_trace(seq, ref, 8, 4, parasail.dnafull)
        counts = defaultdict(int)

        q_coverage = len(alignment.traceback.query) / len(seq)
        r_coverage = len(alignment.traceback.ref) / len(ref)

        if r_coverage < min_coverage:
            return 0.0

        _, cigar = self.parasail_to_sam(alignment, seq)

        for count, op  in re.findall(self.split_cigar, cigar):
            counts[op] += int(count)

        if balanced:
            accuracy = (counts['='] - counts['I']) / (counts['='] + counts['X'] + counts['D'])
        else:
            accuracy = counts['='] / (counts['='] + counts['I'] + counts['X'] + counts['D'])
        return accuracy * 100


    def parasail_to_sam(self, result, seq):
        # From https://github.com/nanoporetech/bonito/blob/655feea4bca17feb77957c7f8be5077502292bcf/bonito/util.py#L321
        """
        Extract reference start and sam compatible cigar string.

        :param result: parasail alignment result.
        :param seq: query sequence.

        :returns: reference start coordinate, cigar string.
        """
        cigstr = result.cigar.decode.decode()
        first = re.search(self.split_cigar, cigstr)

        first_count, first_op = first.groups()
        prefix = first.group()
        rstart = result.cigar.beg_ref
        cliplen = result.cigar.beg_query

        clip = '' if cliplen == 0 else '{}S'.format(cliplen)
        if first_op == 'I':
            pre = '{}S'.format(int(first_count) + cliplen)
        elif first_op == 'D':
            pre = clip
            rstart = int(first_count)
        else:
            pre = '{}{}'.format(clip, prefix)

        mid = cigstr[len(prefix):]
        end_clip = len(seq) - result.end_query - 1
        suf = '{}S'.format(end_clip) if end_clip > 0 else ''
        new_cigstr = ''.join((pre, mid, suf))
        return rstart, new_cigstr        