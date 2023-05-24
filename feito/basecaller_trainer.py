# Trainer class was inspired from https://github.com/nanoporetech/bonito/blob/master/bonito/training.py
import torch
from tqdm import tqdm
from typing import Optional, Callable, List

_Callbacks=Optional[List[Callable]]
_Epoch=Optional[int]

class BasecallerTrainer:
    "Training and Validation of a model"

    def __init__(self, model, device, train_loader, validation_loader, criterion, optimizer, callbacks: _Callbacks):
        self.model=model.to(device) 
        self.device=device
        self.train_loader=train_loader
        self.validation_loader=validation_loader
        self.optimizer=optimizer
        self.criterion=criterion
        self.callbacks=callbacks

        # for callbacks 
        

    def fit(self, epochs):
        # track best loss validation
        best_loss = float("inf")

        for t in range(epochs):
            epoch=t+1
            train_loss = self.train_one_epoch(epoch)    # returns a float
            val_loss   = self.validate_one_epoch(epoch) # returns a float
            print("Validation Loss", val_loss)

            # callbacks
            for callback in self.callbacks:
                
                if callback.__class__.__name__ == "CSVLogger":
                    print(f"calling callback {callback.__class__.__name__}")
                    callback()
                elif callback.__class__.__name__ == "ModelCheckpoint":
                    print(f"calling callback {callback.__class__.__name__}")
                    best_loss = callback(model=self.model, current_loss=val_loss, best_loss=best_loss, epoch=epoch)
        
        print("Done!")



    # Training
    def train_one_batch(self, batch):
        self.optimizer.zero_grad()
        X, y, output_len, target_len = (x.to(self.device) for x in batch)

        # Compute prediction error
        preds  = self.model(X)
        losses = self.criterion(preds, y, output_len, target_len)

        # Backpropagation
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        
        return losses

    def train_one_epoch(self, epoch: int):
        self.model.train()
        n_batches=len(self.train_loader)

        with tqdm(total=n_batches, leave=True, ncols=100, bar_format='{l_bar}{bar}| [{elapsed}{postfix}]') as progress_bar:

            for n_batch, batch in enumerate(self.train_loader):
                
                # Description for progress bar
                progress_bar.set_description(f"Training Epoch: {epoch} | Batch: {n_batch+1}/{n_batches}")


                losses=self.train_one_batch(batch)

                progress_bar.set_postfix(loss='%.4f' % losses)
                progress_bar.update(1)

            # TODO: add callbacks
        
        return losses.item()
    
    # Validation
    def validate_one_batch(self, batch):
        # TODO: validate one batch
        # https://github.com/nanoporetech/bonito/blob/655feea4bca17feb77957c7f8be5077502292bcf/bonito/training.py#L185
        self.model.eval()
        X, y, output_len, target_len = (x.to(self.device) for x in batch)
        preds  = self.model(X)
        losses = self.criterion(preds, y, output_len, target_len)
        
        return losses  
        
    def validate_one_epoch(self, epoch):
        # TODO: validate epoch
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
                    losses = [loss for loss in losses if loss < float("inf")] # FIXME: remove this, it's just to try ModelCheckpoint
                    progress_bar.set_postfix(loss='%.4f' % loss)
                    progress_bar.update(1)
        
        return torch.Tensor(losses).mean().item() 
            

    # to consider accuracies of alinged reads in the validation step 
    # https://github.com/nanoporetech/bonito/blob/655feea4bca17feb77957c7f8be5077502292bcf/bonito/util.py#L354
    # Rodan also has a code to consider a sam file and a reference but is not used in training or validation, they only computes loss
    # https://github.com/biodlab/RODAN/blob/master/accuracy.py

