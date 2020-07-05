import torch

from .thop import profile, clever_format


@torch.no_grad()
def flops(model, inp):
    """
    Estimation of model FLOPs.
    :param model: PyTorch model of which evaluate FLOPs.
    :param inp: Dummy input to feed to the model.
    :return: Esimated FLOPs.
    """
    total_ops, _ = profile(model, inp, verbose=False)
    flops = clever_format([total_ops], "%.3f")
    return flops
