import torch
import numpy as np

def ctc_label_smoothing_loss(log_probs, targets, lengths, weights):
    # TODO: fill docs
    """
        Check https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html for details about the input

        CTC label smoothing loss function
        from Bonito 
        https://github.com/nanoporetech/bonito
        Oxord Nanopore Technologies, Ltd. Public License Version 1.0
        See LICENSE.txt in the bonito repository
    Args:
        log_probs (_type_): tensor of shape [timesteps, batch, channels]
        targets (_type_): tensor of shape [timesteps, batch, channels]
        lengths (_type_): _description_
        weights (_type_): _description_

    Returns:
        dict: _description_
    """    
    T, N, C = log_probs.shape # T: input length, N: batch size, C: number of classes (including blank)
    log_probs_lengths = torch.full(size=(N, ), fill_value=T, dtype=torch.int64)
    loss = torch.nn.functional.ctc_loss(log_probs.to(torch.float32), targets, log_probs_lengths, lengths, reduction='mean', zero_infinity=True)
    label_smoothing_loss = -((log_probs * weights.to(log_probs.device)).mean())
    return {'loss': loss + label_smoothing_loss, 'ctc_loss': loss, 'label_smooth_loss': label_smoothing_loss}