#!/usr/bin/bash
# primary libraries
import argparse
from rich_argparse import RichHelpFormatter
import logging # TODO: add loggings
from pathlib import Path
# ----

def main(args):

    # torch
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader # load batches to the network

    # feito 
    from api import Trainer
    from models import SimpleNet, Rodan
    from loss_functions import ctc_label_smoothing_loss
    from dataloaders import DatasetONT
    from callbacks import CSVLogger, ModelCheckpoint

    ## get input parameters
    # datasets
    PATH_SET=args.path_set
    BATCH_SIZE=args.batch_size
    NUM_WORKERS=args.num_workers
    # training
    MODEL=args.model
    DEVICE=args.device 
    # OUTFILE_DEBUG=args.outfile_debug   
    PATH_CHECKPOINT=args.path_checkpoint

    print(PATH_SET, BATCH_SIZE, MODEL, DEVICE)

    if DEVICE is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: 
        device = DEVICE
    print("Device" , device)

    # network to use
    model=eval(f"{MODEL}()")
    model_output_len = model.output_len # another way to obtain the output of the model https://github.com/biodlab/RODAN/blob/029f7d5eb31b11b53537f13164bfedee0c0786e4/model.py#L317
    try:
        checkpoint = torch.load(PATH_CHECKPOINT)["weights"]
    except:
        checkpoint = torch.load(PATH_CHECKPOINT)
    # #
    # model.eval()
    # with torch.no_grad():
    #     fakedata = torch.rand((1, 1, config.seqlen)) config.seqlen=4096
    #     fakeout = model.forward(fakedata.to(device))
    #     model_output_len = fakeout.shape[0]
    # #
    
    loss_fn = ctc_label_smoothing_loss #  nn.CTCLoss() #  
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

    # dataset
    dataset = DatasetONT(recfile=PATH_SET, output_network_len=model_output_len)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # Callbacks
    # csv_logger=CSVLogger(list_vars=["epoch","train_loss","val_loss"], out_file=OUTFILE_DEBUG, overwrite=True)
    # model_checkpoint = ModelCheckpoint(DIRPATH_CHECKPOINT)

    # Trainer
    trainer=Trainer(
        model=model,
        device=device,
        train_loader=None,
        validation_loader=dataloader,
        criterion=loss_fn,
        optimizer=optimizer,
        callbacks=None,
        checkpoint=checkpoint,
    )

    # fit the model
    loss = trainer.validate_one_epoch(epoch=0)

    return loss 

if __name__=="__main__":

    # Command line options
    parser = argparse.ArgumentParser(
        description="Debug basecaller", 
        formatter_class=RichHelpFormatter
    )
    # datasets
    parser.add_argument("--path-set", help="Path to hdf5 file with split reads to evaluate", type=str, dest="path_set")
    parser.add_argument("--batch-size", help="Number of elements in each batch. Default 16", type=int, dest="batch_size", default=16)
    parser.add_argument("--num-workers", help="Number of workers to be used by Pytorch DataLoader class. Default 4", type=int, dest="num_workers", default=4)
    # inference
    parser.add_argument("--model", help="Name of the model. Options: 'SimpleNet', 'Rodan'. Default 'SimpleNet'", type=str, dest="model", default="SimpleNet")
    parser.add_argument("--device", help="Options: 'cpu' or 'cuda'. Default None, in this case it will try to use 'cuda' if available.", type=str, dest="device", default=None)
    parser.add_argument("--checkpoint", help="path to checkpoint to load weights", type=str, dest="path_checkpoint")
    # callbacks
    # parser.add_argument("--outfile-debug", help="File to store loss per batch. Default 'output/debug/loss-per-batch.csv'", type=str, dest="outfile_debug", default="output/debug/loss-per-batch.csv")
    args = parser.parse_args()
    
    loss = main(args)
    print(loss)