import torch
from torch.nn.functional import pad
import random 
import numpy as np

def gaussian_noise(signal, label, prob=0.5):
    """apply gaussian noise to signal signal with probability <prob>

    Args:
        signal (_type_): _description_
        label (_type_): _description_
        prob (float, optional): probability to . Defaults to 0.5.

    Returns:
        tuple: (signal, label) in reverse order
    """   
    if random.random() < prob:
        
        # Define the mean and standard deviation of the Gaussian noise
        mean = 0
        std = 1

        # Create a tensor of the same size as the original tensor with random noise
        noise = torch.tensor(np.random.normal(mean, std, signal.size()), dtype=torch.float)
        gn_signal = signal + noise
        
        return gn_signal, label
    else:
        return signal, label