import os

import torch
from EIDOSearch.dataloaders import get_dataloader
from EIDOSearch.models.architectures import LeNet300, LeNet5, resnet32
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from torchvision import models as torchvision_models

from config import LOGS_ROOT, LAYERS

from utilities.custom_models import LeNet5_8_16, LeNet5_16_32, LeNet300_32


def get_tb_writer(args):
    return SummaryWriter(os.path.join(LOGS_ROOT, args.name, "tb"))


def get_dataloaders(args):
    if args.valid_size > 0:
        train_loader, valid_loader, test_loader = get_dataloader(
            args.dataset, args.data_dir, args.batch_size, args.valid_size, True, 4, True, args.seed
        )
    else:
        train_loader, test_loader = get_dataloader(
            args.dataset, args.data_dir, args.batch_size, args.valid_size, True, 4, True, args.seed
        )
        valid_loader = None
    
    return train_loader, valid_loader, test_loader


def get_model(args):
    model = _load(args.model)
    if args.ckp_path is not None:
        print("Loading model dictionary from: {}".format(args.ckp_path))
        model.load_state_dict(torch.load(args.ckp_path, map_location="cpu"))
    
    device = torch.device(args.device)
    model.to(device)
    
    os.makedirs(os.path.join(LOGS_ROOT, args.name, "models"), exist_ok=True)
    
    return model


def get_optimizers(args, model):
    pytorch_optimizer = _get_pytorch_optimizer(args, model)
    sensitivity_optimizer = _get_sensitivity_optimizer(args, model)
    
    return pytorch_optimizer, sensitivity_optimizer


def _load(model):
    if model == "lenet300":
        return LeNet300()
    if model == "lenet5":
        return LeNet5()
    if model == "alexnet":
        return torchvision_models.alexnet(True)
    if model == "resnet32":
        return resnet32("A")
    if model == "resnet18":
        return torchvision_models.resnet18(True)
    if model == "resnet50":
        return torchvision_models.resnet50(True)
    if model == "resnet101":
        return torchvision_models.resnet101(True)
    if model == "lenet5_8_16":
        return LeNet5_8_16()
    if model == "lenet5_16_32":
        return LeNet5_16_32()
    if model == "lenet300_32":
        return LeNet300_32()


def _get_pytorch_optimizer(args, model):
    pytorch_optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd,
                            nesterov=args.nesterov)
    
    return pytorch_optimizer


def _get_sensitivity_optimizer(args, model):
    sensitivity_optimizer = None
    
    if args.sensitivity is not None:
        if args.sensitivity == "lobster":
            from EIDOSearch.pruning.sensitivity import LOBSTER
            sensitivity_optimizer = LOBSTER(model, args.lmbda, LAYERS)
        elif args.sensitivity == "neuron-lobster":
            from EIDOSearch.pruning.sensitivity import NeuronLOBSTER
            sensitivity_optimizer = NeuronLOBSTER(model, args.lmbda, LAYERS)
        elif args.sensitivity == "serene":
            from EIDOSearch.pruning.sensitivity import SERENE
            sensitivity_optimizer = SERENE(model, args.serene_type, args.lmbda, args.serene_alpha, LAYERS)
    
    return sensitivity_optimizer


