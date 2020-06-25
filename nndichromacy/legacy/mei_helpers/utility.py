import torch


def cumstom_initial_guess(*args, mean=0, std=1, device="cuda"):
    return torch.empty(*[args]).normal_(mean=mean, std=std).to(device)
