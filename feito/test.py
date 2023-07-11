#!/usr/bin/bash

# primary libraries
import argparse
from rich_argparse import RichHelpFormatter
import logging # TODO: add loggings
from pathlib import Path

# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader # load batches to the network

# feito 
from api import Tester
from models import SimpleNet, Rodan
from dataloaders import DatasetONT
from callbacks import CSVLogger, ModelCheckpoint

# ---- 

def main(args):

    PATH_TEST=args.path_test
    BATCH_SIZE=args.batch_size
    MODEL=args.model
    DEVICE=args.device
    PATH_CHECKPOINT=args.path_checkpoint
    PATH_FASTA=args.path_fasta
    RNA=args.rna
    USE_VITERBI=args.use_viterbi
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
    # model.load_state_dict(PATH_CHECKPOINT)
    if device == "cpu":
        model.load_state_dict(torch.load(PATH_CHECKPOINT, map_location=torch.device('cpu')))
    else: 
        model.load_state_dict(torch.load(PATH_CHECKPOINT))
    model.eval()
    
    # dataset
    dataset_test = DatasetONT(recfile=PATH_TEST, output_network_len=model_output_len)
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    tester=Tester(
        model=model, 
        device=device,
        test_loader=dataloader_test,
        path_fasta=PATH_FASTA,
        rna=RNA,
        use_viterbi=USE_VITERBI
    )

    accuracy = tester(return_basecalled_signals=False)
    
    return accuracy

if __name__=="__main__":
    
    # Command line options
    parser = argparse.ArgumentParser(
        description="Test basecaller", 
        formatter_class=RichHelpFormatter
    )
    # dataset
    parser.add_argument("--path-test", help="Path to hdf5 file with training dataset", type=str, dest="path_test")
    parser.add_argument("--num-workers", help="Number of workers to be used by Pytorch DataLoader class. Default 4", type=int, dest="num_workers", default=4)
    # testing
    parser.add_argument("--batch-size", help="Number of elements in each batch. Default 16", type=int, dest="batch_size", default=16)
    parser.add_argument("--model", help="Name of the model. Options: 'SimpleNet', 'Rodan'", type=str, dest="model", default="SimpleNet")
    parser.add_argument("--device", help="cpu or gpu", type=str, dest="device", default=None)
    parser.add_argument("--path-checkpoint", help="path to checkpoint to be used with the model", type=str, dest="path_checkpoint")
    parser.add_argument("--path-fasta", help="file to save basecalled signals. If not provided only accuracy will be returned", default=None, type=str, dest="path_fasta")
    parser.add_argument("--rna", help="Wheter to use RNA or DNA alphabet. Default: True", type=bool, default=True, dest="rna")
    parser.add_argument("--use-viterbi", help="Use Viterbi Search for basecalling, or Beam Search. Default: True", type=bool, default=True, dest="use_viterbi")
    args = parser.parse_args()
    
    accuracy = main(args)
    print(accuracy)
