import os
from copy import deepcopy

import torch

from config import LOGS_ROOT


@torch.no_grad()
def log_statistics(args, epoch, model, pruning_stat, train_performance, valid_performance, test_performance, lr,
                   top_cr, top_acc, cr_data):
    print_epoch_stat(epoch, pruning_stat, train_performance, valid_performance, test_performance)
    
    if pruning_stat["network_param_ratio"] > top_cr:
        top_cr = pruning_stat["network_param_ratio"]
        top_acc = 0
        
        # Print data of previous CR
        print_data(args, model, cr_data)
    
    if valid_performance[0] > top_acc:
        top_acc = valid_performance[0]
        cr_data = {
            "epoch":             epoch,
            "valid_performance": valid_performance,
            "test_performance":  test_performance,
            "pruning_stat":      pruning_stat,
            "lr":                lr,
            "model":             deepcopy(model)
        }
    
    return top_cr, top_acc, cr_data


def print_epoch_stat(epoch, pruning_stat, train_performance, valid_performance, test_performance):
    print("###########" + "#" * len(str(epoch)))
    print("# EPOCH: {} #".format(epoch))
    print("###########" + "#\n" * len(str(epoch)))
    
    print("-- Performance --")
    print("Train: Top-1 {:.2f}, Top-5 {:.2f}, Loss {:.2f}".format(
        train_performance[0], train_performance[1], train_performance[2]))
    print("Validation: Top-1 {:.2f}, Top-5 {:.2f}, Loss {:.2f}".format(
        valid_performance[0], valid_performance[1], valid_performance[2]))
    print("Test: Top-1 {:.2f}, Top-5 {:.2f}, Loss {:.2f}\n".format(
        test_performance[0], test_performance[1], test_performance[2]))
        
    print("-- Architecture --")
    print("Remaining neurons: {}%".format(pruning_stat["network_neuron_non_zero_perc"]))
    print("Neurons CR: {}".format(pruning_stat["network_neuron_ratio"]))
    print("Remaining parameterts: {}%".format(pruning_stat["network_param_non_zero_perc"]))
    print("Parameters CR: {}".format(pruning_stat["network_param_ratio"]))
    

def print_data(args, model, cr_data):
    with open(os.path.join(LOGS_ROOT, args.name, "log.txt"), "a") as cr_file:
        try:
            cr_file.write("Epoch: {}\n".format(cr_data["epoch"]))
            cr_file.write("Validation Error: {}\n".format(100 - cr_data["valid_performance"][0]))
            cr_file.write("Test Error: {}\n".format(100 - cr_data["test_performance"][0]))
            cr_file.write("Neurons CR: {}\n".format(cr_data["pruning_stat"]["network_neuron_ratio"]))
            cr_file.write("Remaining neurons: {}%\n".format(cr_data["pruning_stat"]["network_neuron_non_zero_perc"]))
            cr_file.write("Parameters CR: {}\n".format(cr_data["pruning_stat"]["network_param_ratio"]))
            cr_file.write("Remaining parameters: {}%\n".format(cr_data["pruning_stat"]["network_param_non_zero_perc"]))
            cr_file.write("Learning Rate: {}\n".format(cr_data["lr"]))
            print("\n")
            
            torch.save(model.state_dict(),
                       os.path.join(LOGS_ROOT, args.name, "models", "{}.pt".format(cr_data["epoch"])))
        except:
            pass
