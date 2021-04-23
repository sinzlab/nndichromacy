import torch


def gauss_loss(output, responses, mean=False):
    loss = - torch.sum((output - responses)**2) if mean is False else - torch.mean((output - responses)**2)
    return loss