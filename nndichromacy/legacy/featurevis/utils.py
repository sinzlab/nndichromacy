import torch


def varargin(f):
    """Decorator to make a function able to ignore named parameters not declared in its
     definition.

    Arguments:
        f (function): Original function.

    Usage:
            @varargin
            def my_f(x):
                # ...
        is equivalent to
            def my_f(x, **kwargs):
                #...
        Using the decorator is recommended because it makes it explicit that my_f won't
        use arguments received in the kwargs dictionary.
    """
    import inspect
    import functools

    # Find the name of parameters expected by f
    f_params = inspect.signature(f).parameters.values()
    param_names = [p.name for p in f_params]  # name of parameters expected by f
    receives_kwargs = any(
        [p.kind == inspect.Parameter.VAR_KEYWORD for p in f_params]
    )  # f receives a dictionary of **kwargs

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if not receives_kwargs:
            # Ignore named parameters not expected by f
            kwargs = {k: kwargs[k] for k in kwargs.keys() if k in param_names}
        return f(*args, **kwargs)

    return wrapper


class Compose:
    """Chain a set of operations into a single function.

    Each function must receive one positional argument and any number of keyword
    arguments. Each function is called with the output of the previous one (as its
    positional argument) and all keyword arguments of the original call.

    Arguments:
        operations (list): List of functions.
    """

    def __init__(self, operations):
        self.operations = operations

    def __call__(self, x, **kwargs):
        if len(self.operations) == 0:
            out = None
        else:
            out = self.operations[0](x, **kwargs)
            for op in self.operations[1:]:
                out = op(out, **kwargs)

        return out

    def __getitem__(self, item):
        return self.operations[item]


class Combine:
    """Applies different operations to an input and combines its output.

    Arguments:
        operations (list): List of operations
        combine_op (function): Function used to combine the results of all the operations.
    """

    def __init__(self, operations, combine_op=torch.sum):
        self.operations = operations
        self.combine_op = combine_op

    def __call__(self, *args, **kwargs):
        if len(self.operations) == 0:
            return
        else:
            results = [op(*args, **kwargs) for op in self.operations]
            return self.combine_op(torch.stack(results, dim=0))

    def __getitem__(self, item):
        return self.operations[item]
