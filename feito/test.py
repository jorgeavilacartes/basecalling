# primary libraries
import argparse
import logging
from pathlib import Path

# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader # load batches to the network

# feito 
from basecaller_tester import BasecallerTester as Tester
from models import SimpleNet, Rodan
from dataloaders.dataloader import DatasetONT
from callbacks import CSVLogger, ModelCheckpoint
# ---- 

def main(args):

    PATH_TEST=args.path_test
    BATCH_SIZE=args.batch_size
    MODEL=args.model
    DEVICE=args.device
    PATH_CHECKPOINT=args.path_checkpoint

    if DEVICE is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: 
        device = DEVICE
    print("Device" , device)

    model=eval(f"{MODEL}()")
    model_output_len = model.output_len

    # dataset
    dataset_test = DatasetONT(recfile=PATH_TEST, output_network_len=model_output_len)
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)
    
    # load weights
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#load
    model.load_state_dict(PATH_CHECKPOINT)

    tester=Tester(
        model=model, 
        device=device,
        test_loader=dataloader_test,
    )

    # inference


if __name__=="__main__":
    
    # Command line options
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--path-test", help="Path to hdf5 file with training dataset", type=str, dest="path_train")
    # testing
    parser.add_argument("--batch-size", help="Number of elements in each batch", type=int, dest="batch_size", default=16)
    parser.add_argument("--model", help="Name of the model. Options: 'SimpleNet', 'Rodan'", type=str, dest="model", default="SimpleNet")
    parser.add_argument("--device", help="cpu or gpu", type=str, dest="device", default=None)
    parser.add_argument("--path-checkpoint", help="path to checkpoint to be used with the model", type=str, dest="path_checkpoint")
    args = parser.parse_args()
    
    main(args)

