# https://github.com/Lyken17/pytorch-OpCounter
# TODO refactor and simplify
import torch

from .profile import profile, profile_origin
from .utils import clever_format

default_dtype = torch.float64
