# Trainer class was inspired from https://github.com/nanoporetech/bonito/blob/master/bonito/training.py
import torch
from tqdm import tqdm
from typing import Optional, Callable, List
# from feito.callbacks import CSVLogger

_Callbacks=Optional[List[Callable]]
_Epoch=Optional[int]

# csv_logger=CSVLogger(list_vars=["train_loss","val_loss"], out_file="output/training/metrics.csv", overwrite=False)

class BasecallerTrainer:

    def __init__(self, model, device, train_loader, validation_loader, criterion, optimizer, callbacks: _Callbacks):
        self.model=model  
        self.device=device
        self.train_loader=train_loader
        self.validation_loader=validation_loader
        self.optimizer=optimizer
        self.criterion=criterion
        self.callbacks=callbacks

    def fit(self, epochs):
        for t in range(epochs):
            epoch=t+1
            train_loss = self.train_one_epoch(epoch)
            val_loss   = self.validate_one_epoch(epoch)

            # callbacks
            for callback in self.callbacks:
                print(f"calling callback {callback.__class__.__name__}")
                callback()

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

                    progress_bar.set_postfix(loss='%.4f' % loss)
                    progress_bar.update(1)
        
        return torch.Tensor(losses).mean().item() 
            

    # to consider accuracies of alinged reads in the validation step 
    # https://github.com/nanoporetech/bonito/blob/655feea4bca17feb77957c7f8be5077502292bcf/bonito/util.py#L354
    # Rodan also has a code to consider a sam file and a reference but is not used in training or validation, they only computes loss
    # https://github.com/biodlab/RODAN/blob/master/accuracy.py

