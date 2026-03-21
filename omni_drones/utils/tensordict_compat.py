# Compatibility shim for tensordict.nn.make_functional
# which was removed in tensordict >= 0.11
#
# This provides a replacement that:
# 1. Extracts nn.Module parameters into a TensorDict
# 2. Replaces module parameters with meta tensors so vmap works
# 3. Patches the module's forward to accept params as second arg

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.func import functional_call


def make_functional(module: nn.Module) -> TensorDict:
    """Extract parameters from an nn.Module and make it functional.

    Replacement for the removed tensordict.nn.make_functional.
    Returns a TensorDict of parameters and patches the module so that
    calling module(input, params_td) uses functional_call under the hood.

    This is needed for vmap compatibility where the module is vmapped
    over both inputs and parameters.
    """
    # Extract parameters as a TensorDict
    params = {}
    param_names = []
    for name, param in module.named_parameters():
        params[name] = param.data.clone()
        param_names.append(name)

    td = TensorDict(params, batch_size=[])

    # Replace module params with empty meta tensors so vmap doesn't
    # complain about non-batched parameters
    for name in param_names:
        parts = name.split('.')
        obj = module
        for part in parts[:-1]:
            obj = getattr(obj, part)
        delattr(obj, parts[-1])
        obj.register_buffer(parts[-1], torch.empty(0))

    # Patch the module's __call__ to accept params TensorDict as second arg
    _orig_forward = module.forward

    def _functional_forward(input_data, params_td=None):
        if params_td is not None:
            # Convert TensorDict to dict for functional_call
            param_dict = {k: v for k, v in params_td.items()}
            return functional_call(module, param_dict, (input_data,),
                                   kwargs={}, strict=False)
        return _orig_forward(input_data)

    module.forward = _functional_forward

    return td
