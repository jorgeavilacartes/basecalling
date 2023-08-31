# Trainer class was inspired from https://github.com/nanoporetech/bonito/blob/master/bonito/training.py
import sys
import torch
from tqdm import tqdm
from typing import Optional, Callable, List

_Callbacks=Optional[List[Callable]]
_Epoch=Optional[int]

import logging
logging.basicConfig(level=logging.INFO,
                    format='[FEITO-trainer] - %(asctime)s. %(message)s',
                    datefmt='%Y-%m-%d@%H:%M:%S')

class BasecallerTrainer:
    "Training and Validation of a model"

    def __init__(self, model, device, train_loader, validation_loader, criterion, optimizer, callbacks: _Callbacks, checkpoint = None):
        self.model=model.to(device)
        self.device=device
        self.train_loader=train_loader
        self.validation_loader=validation_loader
        self.optimizer=optimizer
        self.criterion=criterion
        self.callbacks=[callbacks] if not isinstance(callbacks,list) else callbacks

        if checkpoint:
            try:
                self.model.load_state_dict(checkpoint)
            except:
                self.model.load_state_dict(checkpoint["weights"])
                

    def fit(self, epochs):
        # track best loss validation
        best_loss = float("inf")

        for t in range(epochs):
            epoch=t+1
            train_loss = self.train_one_epoch(epoch)    # returns a float # TODO: return a dictionary with different losses
            val_loss   = self.validate_one_epoch(epoch) # returns a float
            
            logging.info("Training Loss", train_loss)
            logging.info("Validation Loss", val_loss)
            
            # callbacks
            for callback in self.callbacks:
                
                if callback.__class__.__name__ == "CSVLogger":
                    logging.info(f"calling callback {callback.__class__.__name__}, epoch={epoch}")
                    callback()
                elif callback.__class__.__name__ == "ModelCheckpoint":
                    logging.info(f"calling callback {callback.__class__.__name__}, epoch={epoch}")
                    best_loss = callback(model=self.model, optimizer=self.optimizer, current_loss=val_loss, best_loss=best_loss, epoch=epoch)
        
        print("Done!")

    # Training
    def train_one_batch(self, batch):
        self.optimizer.zero_grad()
        X, y, output_len, target_len = (x.to(self.device) for x in batch)

        # Compute prediction error
        preds  = self.model(X)
        # output_len = [preds.shape[0] for _ in range(preds.shape[1])]
        # loss = self.criterion(preds, y, output_len, target_len) # ctc loss
        loss = self.criterion(preds, y, target_len) # ctc_smooth_smoothing

        if loss.item() == float("inf"):
            # print(torch.nn.CTCLoss(reduction="none")(preds, y, output_len, target_len))
            print(X.shape, preds.shape, y.shape, output_len, target_len)
            print(y)
            print(y[0])
            sys.exit()

        # Backpropagation
        loss.backward()
        self.optimizer.step()
        
        return loss

    def train_one_epoch(self, epoch: int):
        #TODO: print average loss function of batches
        self.model.train()
        n_batches=len(self.train_loader)
        losses = []

        with tqdm(total=n_batches, leave=True, ncols=100, bar_format='{l_bar}{bar}| [{elapsed}{postfix}]') as progress_bar:

            for n_batch, batch in enumerate(self.train_loader):
                
                # Description for progress bar
                progress_bar.set_description(f"Training Epoch: {epoch} | Batch: {n_batch+1}/{n_batches}")

                # compute loss for the batch
                loss=self.train_one_batch(batch)
                losses.append(loss.item())

                # show current average loss
                current_avg_loss = torch.Tensor(losses).mean().item()
                progress_bar.set_postfix(train_loss='%.4f' % current_avg_loss)
                progress_bar.update(1)

        return current_avg_loss
    
    # Validation
    def validate_one_batch(self, batch):
        # https://github.com/nanoporetech/bonito/blob/655feea4bca17feb77957c7f8be5077502292bcf/bonito/training.py#L185
        self.model.eval()
        X, y, output_len, target_len = (x.to(self.device) for x in batch)
        preds  = self.model(X)
        # loss = self.criterion(preds, y, output_len, target_len)
        loss = self.criterion(preds, y, target_len) # ctc_smooth_smoothing
        
        return loss  
        
    def validate_one_epoch(self, epoch):
        # https://github.com/nanoporetech/bonito/blob/655feea4bca17feb77957c7f8be5077502292bcf/bonito/training.py#L206
        self.model.eval()
        n_batches=len(self.validation_loader)

        losses = []

        with tqdm(total=n_batches, leave=True, ncols=100, bar_format='{l_bar}{bar}| [{elapsed}{postfix}]') as progress_bar:
            
            with torch.no_grad():
                
                for n_batch, batch in enumerate(self.validation_loader):

                    # Description for progress bar
                    progress_bar.set_description(f"Validate Epoch: {epoch} | Batch: {n_batch+1}/{n_batches}")

                    # compute loss function for the batch
                    loss = self.validate_one_batch(batch)
                    losses.append(loss.item())
                    
                    # show current average loss during training
                    current_avg_loss = torch.Tensor(losses).mean().item()
                    progress_bar.set_postfix(val_loss='%.4f' % current_avg_loss)
                    progress_bar.update(1)
        
        return torch.Tensor(losses).mean().item() 
            

    # to consider accuracies of alinged reads in the validation step 
    # https://github.com/nanoporetech/bonito/blob/655feea4bca17feb77957c7f8be5077502292bcf/bonito/util.py#L354
    # Rodan also has a code to consider a sam file and a reference but is not used in training or validation, they only computes loss
    # https://github.com/biodlab/RODAN/blob/master/accuracy.py

