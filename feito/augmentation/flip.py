import torch
from torch.nn.functional import pad
import random 

def flip(signal, label, prob=0.5):
    """filp horizontally signal and label sequence with probability <prob>

    Args:
        signal (_type_): _description_
        label (_type_): _description_
        prob (float, optional): probability to . Defaults to 0.5.

    Returns:
        tuple: (signal, label) in reverse order
    """    
    # if random.random() > prob:
    #     return signal[::-1], label[::-1]
    # else:
    #     return signal, label


    len_output = 271

    if random.random() < prob:
        non0_label = label[label.nonzero().squeeze().detach()]
        non0_label = torch.flip(non0_label, (-1,))
        
        missing_pos = len_output - len(non0_label)

        rev_seq   = torch.flip(signal, (0,1))
        rev_label = pad(non0_label,(0,missing_pos),value=0)

        return rev_seq, rev_label
    else:
        return signal, label