# !/usr/bin/bash

import logging
logging.basicConfig(level=logging.INFO,
                    format='[FEITO-basecall] - %(asctime)s. %(message)s',
                    datefmt='%Y-%m-%d@%H:%M:%S')

# ---- 
# primary libraries
import argparse
from pathlib import Path
from rich_argparse import RichHelpFormatter
    
def main(args):

    # torch
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader # load batches to the network

    # feito 
    from api import Basecaller
    from models import SimpleNet, Rodan
    from dataloaders import DatasetBasecalling
    from utils.reconstruct_reads import ReconstructReads

    PATH_FAST5=[args.path_fast5] if Path(args.path_fast5).is_file() else  list(Path(args.path_fast5).rglob("*.fast5")) # file or set of reads 
    BATCH_SIZE=args.batch_size
    MODEL=args.model
    DEVICE=args.device
    LEN_SUBSIGNALS=args.len_subsignals
    PATH_CHECKPOINT=args.path_checkpoint
    PATH_FASTA=args.path_fasta # to save basecalled reads (output of the model) 
    PATH_READS=args.path_reads # to save reconstructed reads (full reads: concatenation of basecalled reads)
    RNA=args.rna
    USE_VITERBI=args.use_viterbi
    PATH_INDEX=args.path_index
    NUM_WORKERS=args.num_workers

    if DEVICE is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: 
        device = DEVICE
    print("Device" , device)

    model=eval(f"{MODEL}()")
    model_output_len = model.output_len
    model.to(device)


    # load weights
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#load
    if device == "cpu":
        model.load_state_dict(torch.load(PATH_CHECKPOINT, map_location=torch.device("cpu"))["weights"])
    else: 
        model.load_state_dict(torch.load(PATH_CHECKPOINT)["weights"])
    model.eval()
    
    # dataset
    dataset_basecalling = DatasetBasecalling(
        path_fast5=PATH_FAST5,
        path_save_index=PATH_INDEX,
        len_subsignals=LEN_SUBSIGNALS
        )
    
    basecalling_dataloader = DataLoader(
        dataset_basecalling, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS
        )
    
    basecaller = Basecaller(
        model=model, device=device, 
        dataloader=basecalling_dataloader,
        path_fasta=PATH_FASTA,
        rna=RNA,
        use_viterbi=USE_VITERBI,
    )

    # basecall reads
    print("Basecalling")
    basecaller()
    
    # reconstruct basecalled reads
    print("Reconstruction of reads")
    reconstruct_reads = ReconstructReads(
    path_index=PATH_INDEX,
    path_basecalled_reads=PATH_FASTA,
    path_reconstructed_reads=PATH_READS
    )
    reconstruct_reads()

if __name__=="__main__":
    
    # Command line options
    parser = argparse.ArgumentParser(
        description="Basecall raw signals",
        formatter_class=RichHelpFormatter
    )
    # TODO: add choices to some arguments
    # dataset
    parser.add_argument("--path-fast5", help="Path to single file .fast5 or a directory with .fast5 files", type=str, dest="path_fast5")
    parser.add_argument("--len-subsignals", help="input used by the model. Default 4096", type=int, default=4096, dest="len_subsignals")
    parser.add_argument("--path-index", help="CSV file where to save index of split reads", type=str, default="output/basecalling/index.csv", dest="path_index")
    parser.add_argument("--batch-size", help="Number of elements in each batch. Default 16", type=int, dest="batch_size", default=16)
    parser.add_argument("--num-workers", help="Number of workers to be used by Pytorch DataLoader class. Default 4", type=int, dest="num_workers", default=4)
    
    # basecalling
    parser.add_argument("--model", help="Name of the model. Options: 'SimpleNet', 'Rodan'", type=str, dest="model", default="SimpleNet")
    parser.add_argument("--device", help="cpu or cuda", type=str, dest="device", choices= ["cpu","cuda"], default=None)
    parser.add_argument("--path-checkpoint", help="path to checkpoint to be used with the model", type=str, dest="path_checkpoint")
    parser.add_argument("--path-fasta", help="file to save basecalled signals", default="output/basecalling/basecalled_reads.fa", type=str, dest="path_fasta")
    parser.add_argument("--path-reads", help="file to save reconstructed reads", default="output/basecalling/reconstructed_reads.fa", type=str, dest="path_reads")
    parser.add_argument("--rna", help="use RNA alphabet if provided, otherwise use DNA alphabet", action="store_true", dest="rna")
    parser.add_argument("--use-viterbi", help="Use Viterbi Search for basecalling, otherwise use Beam Search", action="store_true", dest="use_viterbi")
    args = parser.parse_args()

    main(args)