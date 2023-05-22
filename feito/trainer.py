# Trainer class was inspired from https://github.com/nanoporetech/bonito/blob/master/bonito/training.py
from tqdm import tqdm
from typing import Optional, Callable, List

_Callbacks=Optional[List[Callable]]
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

        # TODO: do we need this? used in bonito https://github.com/nanoporetech/bonito/blob/655feea4bca17feb77957c7f8be5077502292bcf/bonito/training.py#L133
        # losses = {
        #             k: (v.item() if losses is None else v.item()+ losses[k])
        #             for k, v in losses.items()
        #         }
        
        return losses

    def train_one_epoch(self, epoch: int):
        self.model.train()
        n_batches=len(self.train_loader)

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
        # https://github.com/nanoporetech/bonito/blob/655feea4bca17feb77957c7f8be5077502292bcf/bonito/training.py#L185
        pass

    def validate_one_epoch(self,):
        # TODO: validate epoch
        # https://github.com/nanoporetech/bonito/blob/655feea4bca17feb77957c7f8be5077502292bcf/bonito/training.py#L206
        pass


    # to consider accuracies of alinged reads in the validation step 
    # https://github.com/nanoporetech/bonito/blob/655feea4bca17feb77957c7f8be5077502292bcf/bonito/util.py#L354
    # Rodan also has a code to consider a sam file and a reference but is not used in training or validation, they only computes loss
    # https://github.com/biodlab/RODAN/blob/master/accuracy.py

