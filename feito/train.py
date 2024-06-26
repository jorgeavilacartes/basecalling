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
    from pytorch_ranger import Ranger

    # feito 
    from api import Trainer
    from models import SimpleNet, Rodan
    from loss_functions import ctc_label_smoothing_loss
    from dataloaders import DatasetONT
    from callbacks import CSVLogger, ModelCheckpoint

    from augmentation.flip import flip
    from augmentation.gaussian_noise import gaussian_noise

    ## get input parameters
    # datasets
    PATH_TRAIN=args.path_train
    PATH_VAL=args.path_val
    EPOCHS=args.epochs
    BATCH_SIZE=args.batch_size
    NUM_WORKERS=args.num_workers
    print(args.augmentation)
    AUGMENTATION=[gaussian_noise]#[eval(aug) for aug in args.augmentation]
    # training
    MODEL=args.model
    DEVICE=args.device 
    PATH_WEIGHTS=args.path_weights
    # callbacks
    OUTFILE_TRAIN_LOGGER=args.outfile_train_logger   
    DIRPATH_CHECKPOINT=args.dirpath_checkpoint

    print(PATH_TRAIN, PATH_VAL, EPOCHS, BATCH_SIZE, MODEL, OUTFILE_TRAIN_LOGGER, DEVICE)

    if DEVICE is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: 
        device = DEVICE
    print("Device" , device)

    # network to use
    model=eval(f"{MODEL}()")
    model_output_len = model.output_len # another way to obtain the output of the model https://github.com/biodlab/RODAN/blob/029f7d5eb31b11b53537f13164bfedee0c0786e4/model.py#L317
    
    # # load weights to be loaded to the model
    # checkpoint = torch.load(PATH_WEIGHTS) if PATH_WEIGHTS else None

    if device == "cpu":
        model.load_state_dict(torch.load(PATH_WEIGHTS, map_location=torch.device('cpu'))["weights"])
    else: 
        model.load_state_dict(torch.load(PATH_WEIGHTS)["weights"])
    # #
    # model.eval()
    # with torch.no_grad():
    #     fakedata = torch.rand((1, 1, config.seqlen)) config.seqlen=4096
    #     fakeout = model.forward(fakedata.to(device))
    #     model_output_len = fakeout.shape[0]
    # #
    
    loss_fn = ctc_label_smoothing_loss # nn.CTCLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, momentum=0.9)
    optimizer = Ranger(model.parameters(), lr=2e-3, weight_decay=0.01)

    # dataset
    dataset_train = DatasetONT(recfile=PATH_TRAIN, output_network_len=model_output_len, augmentations=AUGMENTATION)
    dataset_val   = DatasetONT(recfile=PATH_VAL, output_network_len=model_output_len)

    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Callbacks
    csv_logger=CSVLogger(list_vars=["epoch","train_loss","val_loss"], out_file=OUTFILE_TRAIN_LOGGER, overwrite=True)
    model_checkpoint = ModelCheckpoint(DIRPATH_CHECKPOINT)

    # Trainer
    trainer=Trainer(
        model=model,
        device=device,
        train_loader=dataloader_train,
        validation_loader=dataloader_val,
        criterion=loss_fn,
        optimizer=optimizer,
        callbacks=[csv_logger, model_checkpoint],
        # checkpoint=checkpoint
    )

    # fit the model
    trainer.fit(epochs=EPOCHS)

if __name__=="__main__":

    # Command line options
    parser = argparse.ArgumentParser(
        description="Train basecaller", 
        formatter_class=RichHelpFormatter
    )
    # datasets
    parser.add_argument("--path-train", help="Path to hdf5 file with training dataset", type=str, dest="path_train")
    parser.add_argument("--path-val", help="Path to hdf5 file with validation dataset", type=str, dest="path_val")
    parser.add_argument("--epochs", help="Number of epochs the model will be trained. Default 5", type=int, dest="epochs", default=5)
    parser.add_argument("--batch-size", help="Number of elements in each batch. Default 16", type=int, dest="batch_size", default=16)
    parser.add_argument("--num-workers", help="Number of workers to be used by Pytorch DataLoader class. Default 4", type=int, dest="num_workers", default=4)
    parser.add_argument("--augmentation", help="list of augmentations to apply to signal/label", type=str, dest="augmentation", nargs="+", default=[])
    # training
    parser.add_argument("--model", help="Name of the model. Options: 'SimpleNet', 'Rodan'. Default 'SimpleNet'", type=str, dest="model", default="SimpleNet")
    parser.add_argument("--device", help="Options: 'cpu' or 'cuda'. Default None, in this case it will try to use 'cuda' if available.", type=str, dest="device", default=None)
    parser.add_argument("--path-weights", help="path to weights of a trained model", type=str, dest="path_weights", default=None)
    # callbacks
    parser.add_argument("--outfile-train-logger", help="File to store training and validation loss per epoch. Default 'output/training/metrics.csv'", type=str, dest="outfile_train_logger", default="output/training/metrics.csv")
    parser.add_argument("--dirpath-checkpoint", help="directory where best weights will be saved. Default 'output/training/checkpoints'", type=str, dest="dirpath_checkpoint", default="output/training/checkpoints")
    args = parser.parse_args()
    
    main(args)