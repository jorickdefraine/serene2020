from copy import deepcopy

import torch
from EIDOSearch.evaluation import test_model, architecture_stat
from EIDOSearch.pruning import get_mask_par, get_mask_neur
from EIDOSearch.pruning.thresholding import threshold_scheduler
from torch import nn

from config import LAYERS
from utilities import get_dataloaders, log_statistics, print_data


def train_model_epoch_pruning(args, model, train_loader, valid_loader, test_loader, pytorch_optmizer,
                              sensitivity_optmizer):
    device, loss_function, cross_valid,\
    top_cr, top_acc, cr_data,\
    epochs_count, high_lr, low_lr, current_lr = init_train(args)
    
    # Get threshold scheduler
    TS = threshold_scheduler(model, LAYERS, valid_loader, loss_function, args.twt, args.pwe)
    
    # Epochs
    for epoch in range(args.epochs):
        mask_params, mask_neurons = get_masks(args, model)
        
        # Batches
        for data, target in train_loader:
            model.train()
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            # Perform and update step
            optimizer_steps(args, model, data, target, loss_function, pytorch_optmizer, sensitivity_optmizer,
                            mask_params, mask_neurons)
        
        # Get and save epoch statistics
        valid_performance, top_cr, top_acc, cr_data = get_and_save_statistics(args, epoch, model, loss_function,
                                                                              train_loader, valid_loader, test_loader,
                                                                              pytorch_optmizer, top_cr, top_acc,
                                                                              cr_data)
        
        # Perform pruning step
        pruning_step(args, TS, valid_performance, cross_valid)
        
        if args.lr_cycling:
            epochs_count, current_lr = cycle_lr(epochs_count, args.cycle_up, args.cycle_down,
                                                current_lr, low_lr, high_lr, pytorch_optmizer)
    
    print_data(args, model, cr_data)


def train_model_batch_pruning(args, model, train_loader, valid_loader, test_loader, pytorch_optmizer,
                              sensitivity_optmizer):
    device, loss_function, cross_valid,\
    top_cr, top_acc, cr_data,\
    epochs_count, high_lr, low_lr, current_lr = init_train(args)
    
    # Get threshold scheduler
    TS = threshold_scheduler(model, LAYERS, valid_loader, loss_function, args.twt, args.pwe)
    
    print("Batch pruning with pruning every {} batches and test every {} batches"
          .format(args.prune_iter, args.test_iter))
    
    # Epochs
    for epoch in range(args.epochs):
        
        # Batches
        for batch, (data, target) in enumerate(train_loader):
            mask_params, mask_neurons = get_masks(args, model)
            
            model.train()
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            optimizer_steps(args, model, data, target, loss_function, pytorch_optmizer, sensitivity_optmizer,
                            mask_params, mask_neurons)
            
            # Get and save epoch statistics
            if ((batch + 1) % args.test_iter) == 0:
                valid_performance, top_cr, top_acc, cr_data = get_and_save_statistics(args, epoch, model, loss_function,
                                                                                      train_loader, valid_loader,
                                                                                      test_loader,
                                                                                      pytorch_optmizer, top_cr, top_acc,
                                                                                      cr_data)
            
            # Perform pruning step
            if ((batch + 1) % args.prune_iter) == 0:
                
                # Evaluate model performance for pruning purposes only if this batch is not already a 'test_iter'
                if ((batch + 1) % args.test_iter) != 0:
                    valid_performance = test_model(model, loss_function, valid_loader)
                
                pruning_step(args, TS, valid_performance, cross_valid)
        
        if args.cycle_lr:
            epochs_count, current_lr = cycle_lr(epochs_count, args.cycle_up, args.cycle_down,
                                                current_lr, low_lr, high_lr, pytorch_optmizer)
    
    print_data(args, model, cr_data)


def optimizer_steps(args, model, data, target, loss_function,
                    pytorch_optmizer, sensitivity_optimizer, mask_params, mask_neurons):
    """
    Performs inference and parameters update using both the pytorch optimizer and the sensitivity optimizer
    :param args: Run arguments
    :param model: PyTorch model
    :param data: Model's input
    :param target: Inference target
    :param loss_function: Loss function used to compute the classification loss
    :param pytorch_optmizer: PyTorch optimizer (e.g. SGD)
    :param sensitivity_optimizer: Sensitivity optimizer
    :param mask_params: Dictionary of binary tensors, returned by `get_masks`
    :param mask_neurons: Dictionary of binary tensors, returned by `get_masks`
    """
    # Zero grad, inference, loss computation
    pytorch_optmizer.zero_grad()
    output = model(data)
    loss = loss_function(output, target)
    
    # If the sensitivity optimizer is SERENE with mode 'full', we have to maintain the loss backward graph
    # else we can discard it
    if args.sensitivity == "serene":
        if args.serene_type == "full":
            loss.backward(retain_graph=True)
    
        # If the sensitivity optimizer is SERENE with mode 'lower-bound' we compute the backward propagation from the output
        elif args.serene_type == "lower-bound":
            output.backward(torch.ones_like(output), retain_graph=True)
            tmp_preact = deepcopy(sensitivity_optimizer.preactivations)
            
            pytorch_optmizer.zero_grad()
            loss.backward()
            
            sensitivity_optimizer.preactivations = tmp_preact
            del tmp_preact
    
        sensitivity_optimizer.step(output, [mask_params, mask_neurons])
        
    else:
        loss.backward()
        sensitivity_optimizer.step([mask_params, mask_neurons])

    del output, loss
    
    pytorch_optmizer.step()


def get_and_save_statistics(args, epoch, model, loss_function,
                            train_loader, valid_loader, test_loader, pytorch_optmizer,
                            top_cr, top_acc, cr_data):
    pruning_stat = architecture_stat(model)
    
    train_performance = test_model(model, loss_function, train_loader)
    valid_performance = test_model(model, loss_function, valid_loader)
    test_performance = test_model(model, loss_function, test_loader)
    
    top_cr, top_acc, cr_data = log_statistics(args, epoch, model, pruning_stat, train_performance,
                                              valid_performance,
                                              test_performance, pytorch_optmizer.param_groups[0]['lr'], top_cr,
                                              top_acc, cr_data)
    
    return valid_performance, top_cr, top_acc, cr_data


def pruning_step(args, TS, valid_performance, cross_valid):
    if TS.step(valid_performance[2], args.batch_pruning):
        if cross_valid:
            args.seed += 1
            train_loader, valid_loader, test_loader = get_dataloaders(args)
            
            TS.set_validation_loader(valid_loader)
        return True
    
    return False


def cycle_lr(epochs_count, cycle_up, cycle_down, current_lr, low_lr, high_lr, pytorch_optmizer):
    if epochs_count == cycle_up and current_lr == low_lr:
        for param_group in pytorch_optmizer.param_groups:
            param_group['lr'] = high_lr
        
        current_lr = high_lr
        epochs_count = 1
    
    elif epochs_count == cycle_down and current_lr == high_lr:
        for param_group in pytorch_optmizer.param_groups:
            param_group['lr'] = low_lr
        
        current_lr = low_lr
        epochs_count = 1
    
    else:
        epochs_count += 1
    
    return epochs_count, current_lr


def get_masks(args, model):
    mask_params = get_mask_par(model, LAYERS) if args.mask_params else None
    mask_neurons = get_mask_neur(model, LAYERS) if args.mask_neurons else None
    return mask_params, mask_neurons


def init_train(args):
    device = torch.device(args.device)
    loss_function = nn.CrossEntropyLoss().to(device)
    cross_valid = True
    top_cr = 1
    top_acc = 0
    cr_data = {}
    
    epochs_count = 1
    high_lr = args.lr
    low_lr = args.lr / 10
    current_lr = high_lr
    
    return device, loss_function, cross_valid, top_cr, top_acc, cr_data, epochs_count, high_lr, low_lr, current_lr
