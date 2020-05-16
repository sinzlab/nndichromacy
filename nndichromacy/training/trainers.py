import warnings
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

from mlutils.measures import *
from mlutils import measures as mlmeasures
from mlutils.training import early_stopping, MultipleObjectiveTracker, eval_state, cycle_datasets, Exhauster, LongCycler
from nnfabrik.utility.nn_helpers import set_random_seed

from ..utility import measures
from ..utility.measures import get_correlations, get_poisson_loss


def nnvision_trainer(model, dataloaders, seed, avg_loss=False, scale_loss=True,  # trainer args
                                loss_function='PoissonLoss', stop_function='get_correlations',
                                loss_accum_batch_n=None, device='cuda', verbose=True,
                                interval=1, patience=5, epoch=0, lr_init=0.005,  # early stopping args
                                max_iter=100, maximize=True, tolerance=1e-6,
                                restore_best=True, lr_decay_steps=3,
                                lr_decay_factor=0.3, min_lr=0.0001,  # lr scheduler args
                                cb=None, track_training=False, return_test_score=False, **kwargs):
    """

    Args:
        model:
        dataloaders:
        seed:
        avg_loss:
        scale_loss:
        loss_function:
        stop_function:
        loss_accum_batch_n:
        device:
        verbose:
        interval:
        patience:
        epoch:
        lr_init:
        max_iter:
        maximize:
        tolerance:
        restore_best:
        lr_decay_steps:
        lr_decay_factor:
        min_lr:
        cb:
        track_training:
        **kwargs:

    Returns:

    """

    def full_objective(model, dataloader, data_key, *args):
        """

        Args:
            model:
            dataloader:
            data_key:
            *args:

        Returns:

        """
        loss_scale = np.sqrt(len(dataloader[data_key].dataset) / args[0].shape[0]) if scale_loss else 1.0
        return loss_scale * criterion(model(args[0].to(device), data_key), args[1].to(device)) \
               + model.regularizer(data_key)

    ##### Model training ####################################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()

    criterion = getattr(mlmeasures, loss_function)(avg=avg_loss)
    stop_closure = partial(getattr(measures, stop_function), dataloaders=dataloaders["validation"], device=device, per_neuron=False, avg=True)

    n_iterations = len(LongCycler(dataloaders["train"]))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if maximize else 'min',
                                                           factor=lr_decay_factor, patience=patience,
                                                           threshold=tolerance,
                                                           min_lr=min_lr, verbose=verbose, threshold_mode='abs')

    # set the number of iterations over which you would like to accummulate gradients
    optim_step_count = len(dataloaders["train"].keys()) if loss_accum_batch_n is None else loss_accum_batch_n

    if track_training:
        tracker_dict = dict(correlation=partial(get_correlations(), model=model, dataloaders=dataloaders["validation"], device=device, per_neuron=False),
                            poisson_loss=partial(get_poisson_loss(), model=model, datalaoders=dataloaders["validation"], device=device, per_neuron=False, avg=False))
        if hasattr(model, 'tracked_values'):
            tracker_dict.update(model.tracked_values)
        tracker = MultipleObjectiveTracker(**tracker_dict)
    else:
        tracker = None

    # train over epochs
    for epoch, val_obj in early_stopping(model, stop_closure, interval=interval, patience=patience,
                                         start=epoch, max_iter=max_iter, maximize=maximize,
                                         tolerance=tolerance, restore_best=restore_best, tracker=tracker,
                                         scheduler=scheduler, lr_decay_steps=lr_decay_steps):

        # print the quantities from tracker
        if verbose and tracker is not None:
            print("=======================================")
            for key in tracker.log.keys():
                print(key, tracker.log[key][-1], flush=True)

        # executes callback function if passed in keyword args
        if cb is not None:
            cb()

        # train over batches
        optimizer.zero_grad()
        for batch_no, (data_key, data) in tqdm(enumerate(LongCycler(dataloaders["train"])), total=n_iterations,
                                               desc="Epoch {}".format(epoch)):

            loss = full_objective(model, dataloaders["train"], data_key, *data)
            loss.backward()
            if (batch_no + 1) % optim_step_count == 0:
                optimizer.step()
                optimizer.zero_grad()

    ##### Model evaluation ####################################################################################################
    model.eval()
    tracker.finalize() if track_training else None

    # Compute avg validation and test correlation
    validation_correlation = get_correlations(model, dataloaders["validation"], device=device, as_dict=False, per_neuron=False)
    test_correlation = get_correlations(model, dataloaders["test"], device=device, as_dict=False, per_neuron=False)

    # return the whole tracker output as a dict
    output = {k: v for k, v in tracker.log.items()} if track_training else {}
    output["validation_corr"] = validation_correlation

    score = np.mean(test_correlation) if return_test_score else np.mean(validation_correlation)
    return score, output, model.state_dict()

