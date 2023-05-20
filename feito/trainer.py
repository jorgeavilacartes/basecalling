# Trainer class was inspired from https://github.com/nanoporetech/bonito/blob/master/bonito/training.py
from tqdm import tqdm
from typing import Optional, Callable

_Callbacks=Optional[list[Callable]]
_Epoch=Optional[int]

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
            epoch = t+1
            self.train_one_epoch(epoch)
            # self.validate_one_epoch(epoch)
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

        # losses = {
        #             k: (v.item() if losses is None else v.item()+ losses[k])
        #             for k, v in losses.items()
        #         }
        
        return losses

    def train_one_epoch(self, epoch: int):
        self.model.train()
        n_batches=len(self.train_loader)

        # progress_bar=tqdm(total=n_batches), #leave=True, ncols=100,)
        #                 #   bar_format='{l_bar}{bar}| [{elapsed}{postfix}]')

        with tqdm(total=n_batches, leave=True, ncols=100, bar_format='{l_bar}{bar}| [{elapsed}{postfix}]') as progress_bar:

            for n_batch, batch in enumerate(self.train_loader):

                # Description for progress bar
                if epoch:
                    progress_bar.set_description(f"Epoch: {epoch} | batch: {n_batch}/{n_batches}")
                else:
                    progress_bar.set_description(f"batch {n_batch}/{n_batches}")

                losses=self.train_one_batch(batch)

                progress_bar.set_postfix(loss='%.4f' % losses)
                progress_bar.update(1)

            # TODO: add callbacks

    # Validation
    def validate_one_batch(self,):
        # TODO: validate one batch
        pass

    def validate_one_epoch(self,):
        # TODO: validate epoch
        pass


    # def train(self, dataloader, model, loss_fn, optimizer, epoch: _Epoch):
    #     """
    #     training function
    #     """
    #     device = self.device
    #     size = len(dataloader.dataset) # number of datapoints in the dataset
    #     n_batches = len(dataloader)    # number of batches
    #     model.train() # set model in training mode

    #     with tqdm(total=n_batches, leave=True, ncols=100) as pbar:
            
    #         for batch, (X,y, input_len, target_len) in enumerate(dataloader):
                
    #             # Description for progress bar
    #             if epoch:
    #                 pbar.set_description(f"Epoch: {epoch} | batch: {batch}/{n_batches}")
    #             else:
    #                 pbar.set_description(f"batch {batch}/{n_batches}")
                    
    #             X, y, input_len, target_len = X.to(device), y.to(device), input_len.to(device), target_len.to(device)

    #             # Compute prediction error
    #             pred = model(X)
    #             loss = loss_fn(pred, y, input_lengths=input_len, target_lengths=target_len)

    #             # Backpropagation
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()

    #             if batch % 100 == 0: 
    #                 loss, current = loss.item(), (batch + 1) * len(X)
    #                 # print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    #             # update progress bar
    #             pbar.update(1)
