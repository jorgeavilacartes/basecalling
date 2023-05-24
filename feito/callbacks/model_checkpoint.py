# https://pytorch.org/tutorials/beginner/saving_loading_models.html
import torch

from typing import Union
from pathlib import Path

class ModelCheckpoint:
    # FIXME: check how to use this: "epoch{epoch}-val_loss{val_loss:.2f}" in the call function
    def __init__(self, dirpath: Union[str,Path], filename: str = None ):
        self.dirsave = Path(dirpath)
        self.filename = filename

        # create parent folders for directory path where weights will be saved
        self.dirsave.mkdir(exist_ok=True, parents=True)

    def __call__(self, model, current_loss, best_loss, epoch):
        "Save model weights and biases if loss decreases. Returns best loss so far"
               
        name_model = model.__class__.__name__
        print(current_loss, best_loss, name_model)

        if current_loss < best_loss:
            val_loss = current_loss
            filename = f"{name_model}-epoch{epoch}"
            path_save = self.dirsave.joinpath(filename)
            print(f"saving model for epoch {epoch}")
            torch.save(model.state_dict(), path_save)
        
            return current_loss
        
        return best_loss