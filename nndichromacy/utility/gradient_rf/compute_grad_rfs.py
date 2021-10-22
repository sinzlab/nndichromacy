import numpy as np
import torch
from torch import nn, optim


def compute_RFs(
    model,
    dataloaders,
    data_key,
    selected_channels=None,
    init_img=None,
    neuron_positions=None,
    lr=1,
    model_forward_kwargs=None,
):
    """
    Computes the gradient receptive fields of the neurons through gradients.

    Args:
        model (nn.Module): A model trained on static images predicting neural responses
        dataloaders (dict): A dictionary of dictionaries consiting of
                dataloaders (similar to nnfabrik convention)
        data_key (str): data_key specifying a specific dataset (or session)
        selected_channels (iterable, optional): channels for which you want the gradient RF.
                By default returns the gradient for all channels.
        init_img (nn.Parameter, optional): Image used to compute responses. Defaults to None.
        neuron_positions (iterable, optional): An iterable specifying the postion of neurons
                of interest. Defaults to None.

    Returns:
        np.ndarray: gradient receptive fields computed for specified neurons
    """

    sample_input = next(iter(dataloaders["train"][data_key]))[0]
    _, c, h, w = sample_input.shape
    device = sample_input.device
    selected_channels = (
        selected_channels if selected_channels is not None else np.arange(c)
    )

    init_img = (
        torch.nn.Parameter(init_img.to(device))
        if init_img is not None
        else nn.Parameter(torch.randn(1, c, h, w, device=device))
    )

    model_forward_kwargs = (
        model_forward_kwargs if model_forward_kwargs is not None else dict()
    )

    optimizer = optim.Adam([init_img], lr=lr)
    m = model(init_img, data_key=data_key, **model_forward_kwargs)

    neuron_positions = (
        neuron_positions if neuron_positions is not None else range(m.shape[1])
    )

    grad_RFs = []
    for neuron_position in neuron_positions:
        optimizer.zero_grad()

        neuron_resp = m[0, neuron_position]
        neuron_resp.backward(retain_graph=True)

        grad_RF = init_img.grad.data.cpu().numpy()
        grad_RFs.append(grad_RF[0, selected_channels])

    return np.array(grad_RFs)
