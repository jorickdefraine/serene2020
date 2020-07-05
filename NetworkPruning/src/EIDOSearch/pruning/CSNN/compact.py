import warnings

from .compactor import *
from .creator import *
from .fuser import *


def compact(model, pinned_in=None, pinned_out=None):
    """
    Defines the compact representation (without zeroed-out neurons) of a given model.
    :param model: PyTorch model to compact.
    :param pinned_in: List of layers for which the input neurons shouldn't be processed.
    :param pinned_out:  List of layers for which the output neurons shouldn't be processed.
    :return: Compact PyTorch model.
    """
    warnings.warn("Under development, use at own risk")
    # TODO complete for residuals, refactor, check...

    if pinned_in is None:
        pinned_in = []
    if pinned_out is None:
        pinned_out = []

    fused_model = fuse(model)
    compact_dict = compact_network(fused_model, pinned_in, pinned_out)
    compact_model = build_network(fused_model, compact_dict)

    return compact_model
