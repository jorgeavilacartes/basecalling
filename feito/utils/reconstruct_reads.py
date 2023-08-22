"""
Reconstruct reads from portions of basecalled reads

Needed
- index (for each portion we have the read it cames from)
- basecalled reads in fasta format (header correspond to the index)
"""
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from collections import defaultdict, namedtuple

class ReconstructReads:

    def __init__(self, path_index, path_basecalled_reads, path_reconstructed_reads, reverse_seqs: bool=True):
                 
        self.path_index = path_index
        self.path_basecalled_reads = path_basecalled_reads
        self.path_reconstructed_reads = path_reconstructed_reads
        self.reverse_seqs = reverse_seqs
        self.index_dict = self.create_auxiliar_dict()
        

    def __call__(self,):
        """Reconstruct complete reads from portions of basecalled reads
        Full reads will be saved in the fasta file self.path_reconstructed_reads
        """
        
        reads_by_portions = self.join_portions_reads()

        # reconstructed reads: concatenate portions
        reconstructed_reads = self.reconstruct_reads(reads_by_portions)
        self.reads_to_fasta(reconstructed_reads)

    def load_index(self,): 
        # sep = "," if self.path_index.endswith("csv") else "\t"
        index = pd.read_csv(self.path_index, index_col=False)
        return index

    def create_auxiliar_dict(self):
        id2info={}
        index=self.load_index()
        print(index.head())
        print(index.columns)
        for read_id, idx, subsignal_id in  tqdm(zip(index.read_id, index.index.tolist(), index.subsignal_id.tolist()), total=index.shape[0]):
            id2info[int(idx)] = (read_id, int(subsignal_id))

        return id2info

    def join_portions_reads(self,):        

        # join portion of reads
        reads_by_portions = defaultdict(list)

        with open(self.path_basecalled_reads,"r") as fp:
            reads=SeqIO.parse(fp,"fasta")
            nreads=0
            for read in reads:
                nreads=+1
                if nreads > 100:
                    break
                idx = int(read.description)
                seq = read.seq

                read_id = self.index_dict[idx][0]
                order_portion = self.index_dict[idx][1]

                reads_by_portions[read_id].append((order_portion, seq))

        return reads_by_portions

    def reconstruct_reads(self, reads_by_portions):
        # reconstructed reads: concatenate portions
        reconstructed_reads = {}
        for read_id, lseqs in reads_by_portions.items():

            lseqs = sorted(lseqs, key=lambda t: t[0])
            read = "".join([str(s[1]) for s in lseqs])

            if self.reverse_seqs:
                read = read[::-1]
            reconstructed_reads[read_id] = read
        
        return reconstructed_reads

    def reads_to_fasta(self, reconstructed_reads):
        # save reads to fasta
        with open(self.path_reconstructed_reads, "w") as fp: 
            for readid, read in reconstructed_reads.items():
                fp.write(f">{readid}\n")
                fp.write(read+"\n")