from typing import Union
from pathlib import Path
from fast_ctc_decode import beam_search, viterbi_search


# types
_Path = Union[Path,str]

class BasecallerTester:
    
    def __init__(self, model, device, test_loader, path_fasta: _Path, rna: bool = True, use_viterbi = True):
        self.model=model.to(device) 
        self.device=device
        self.test_loader=test_loader # load signals
        self.path_fasta=path_fasta # save basecalled raw-reads
        self.alphabet="NACGU" if rna else "NACGT"
        self.search_algo=viterbi_search if use_viterbi  else beam_search

    def __call__(self,):
        # TODO: implement tester, considering accuracy

        # 1. generate output of a signal

        # 2. call viterbi or beam search to generate portion of a read
        
        # 3. send to a fasta file
        
        pass

    
    def predict_one_batch(self, batch):
        pass

    def signal_to_read(signal):

        pass