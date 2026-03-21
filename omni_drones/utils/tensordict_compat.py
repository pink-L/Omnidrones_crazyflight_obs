# Compatibility shim for tensordict.nn.make_functional
# which was removed in tensordict >= 0.11
#
# In the original API, make_functional extracted params into a TensorDict
# and patched the module so vmap(module)(inputs, params) works.
#
# Since torch.func.vmap + functional_call is fragile with complex modules,
# we take a simpler approach: just extract params as a TensorDict.
# The calling code (multirotor.py) is patched to use direct tensor ops
# instead of vmap(module).

import torch
import torch.nn as nn
from tensordict import TensorDict


def make_functional(module: nn.Module) -> TensorDict:
    """Extract parameters from an nn.Module into a TensorDict.

    Returns a TensorDict containing clones of all named parameters.
    The module itself is left unchanged (parameters stay as buffers).
    """
    params = {}
    for name, param in module.named_parameters():
        params[name] = param.data.clone()

    # Convert parameters to buffers so they don't show up as module state
    # that vmap would complain about
    param_names = list(params.keys())
    for name in param_names:
        parts = name.split('.')
        obj = module
        for part in parts[:-1]:
            obj = getattr(obj, part)
        data = getattr(obj, parts[-1]).data.clone()
        delattr(obj, parts[-1])
        obj.register_buffer(parts[-1], data)

    module.requires_grad_(False)
    td = TensorDict(params, batch_size=[])
    return td
