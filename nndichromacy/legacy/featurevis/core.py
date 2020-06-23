import warnings

import torch
from torch import optim

from featurevis.exceptions import FeatureVisException


def gradient_ascent(
    f,
    x,
    transform=None,
    regularization=None,
    gradient_f=None,
    post_update=None,
    optim_name="SGD",
    step_size=0.1,
    optim_kwargs={},
    num_iterations=1000,
    save_iters=None,
    print_iters=100,
):
    """ Maximize f(x) via gradient ascent.

    Objective: f(transform(x)) - regularization(transform(x))
    Update: x_{t+1} = post_update(x_{t} + step_size * gradient_f(x_{t}.grad))

    Arguments:
        f (function): Real-valued differentiable function to be optimized
        x (torch.Tensor): Initial guess of the input to optimize.
        transform (function): Differentiable transformation applied to x before sending it
            through the model, e.g., an image generator, jittering, scaling, etc.
        regularization (function): Differentiable regularization term, e.g., natural
            prior, total variation, bilateral filters, etc.
        gradient_f (function): Non-differentiable. Receives the gradient of x and outputs
            a preconditioned gradient, e.g., blurring, masking, etc.
        post_update (function): Non-differentiable. Function applied to x after each
            gradient update, e.g., keep the image norm to some value, blurring, etc.
        optim_name (string): Optimizer to use: SGD or Adam.
        step_size (float): Size of the step size to give every iteration.
        optim_kwargs (dict): Dictionary with kwargs for the optimizer
        num_iterations (int): Number of gradient ascent steps.
        save_iters (None or int): How often to save x. If None, it returns the best x;
            otherwise it saves x after each save_iters iterations.
        print_iters (int): Print some results every print_iters iterations.

    Returns:
        optimal_x (torch.Tensor): x that maximizes the desired function. If save_iters is
            not None, this will be a list of tensors.
        fevals (list): Function evaluations at each iteration. We also evaluate at x_0
            (the original input) so this will have max_iterations + 1 elements.
        reg_terms (list): Value of the regularization term at each iteration. We also
            evaluate at x_0 (the original input) so this will have max_iterations + 1
            elements. Empty if regularization is None.

    Note:
        transform, regularization, gradient_f and post_update receive one positional
        parameter (its input) and the following optional named parameters:
            iteration (int): Current iteration (starts at 1).

        The number of optional parameters may increase so we recommend to write functions
        that receive **kwargs (or use the varargin decorator below) to make sure they will
        still work if we add other optional parameters in the future.
    """
    # Basic checks
    if x.dtype != torch.float32:
        raise ValueError("x must be of torch.float32 dtype")
    x = x.detach().clone()  # to avoid changing original
    x.requires_grad_()

    # Declare optimizer
    if optim_name == "SGD":
        optimizer = optim.SGD([x], lr=step_size, **optim_kwargs)
    elif optim_name == "Adam":
        optimizer = optim.Adam([x], lr=step_size, **optim_kwargs)
    else:
        raise ValueError("Expected optim_name to be 'SGD' or 'Adam'")

    # Run gradient ascent
    fevals = []  # to store function evaluations
    reg_terms = []  # to store regularization function evaluations
    saved_xs = []  # to store xs (ignored if save_iters is None)
    for i in range(1, num_iterations + 1):
        # Zero gradients
        if x.grad is not None:
            x.grad.zero_()

        # Transform input
        transformed_x = x if transform is None else transform(x, iteration=i)

        # f(x)
        feval = f(transformed_x)
        fevals.append(feval.item())

        # Regularization
        if regularization is not None:
            reg_term = regularization(transformed_x, iteration=i)
            reg_terms.append(reg_term.item())
        else:
            reg_term = 0

        # Compute gradient
        (-feval + reg_term).backward()
        if x.grad is None:
            raise FeatureVisException("Gradient did not reach x.")

        # Precondition gradient
        x.grad = x.grad if gradient_f is None else gradient_f(x.grad, iteration=i)
        if (torch.abs(x.grad) < 1e-9).all():
            warnings.warn("Gradient for x is all zero")

        # Gradient ascent step (on x)
        optimizer.step()

        # Cleanup
        if post_update is not None:
            with torch.no_grad():
                x[:] = post_update(x, iteration=i)  # in place so the optimizer still points to the right object

        # Report results
        if i % print_iters == 0:
            feval = feval.item()
            reg_term = reg_term if regularization is None else reg_term.item()
            x_std = x.std().item()
            print("Iter {}: f(x) = {:.2f}, reg(x) = {:.2f}, std(x) = {:.2f}".format(i, feval, reg_term, x_std))

        # Save x
        if save_iters is not None and i % save_iters == 0:
            saved_xs.append(x.detach().clone())

    # Record f(x) and regularization(x) for the final x
    with torch.no_grad():
        transformed_x = x if transform is None else transform(x, iteration=i + 1)

        feval = f(transformed_x)
        fevals.append(feval.item())

        if regularization is not None:
            reg_term = regularization(transformed_x, iteration=i + 1)
            reg_terms.append(reg_term.item())
    print("Final f(x) = {:.2f}".format(fevals[-1]))

    # Set opt_x
    opt_x = x.detach().clone() if save_iters is None else saved_xs

    return opt_x, fevals, reg_terms
