from ...pruning import *


def build_network(pruned, compact_state_dict):
    """
    Define the compact network architecture
    :param pruned: Original pruned model to be compacted
    :param compact_state_dict: Compact state_dict used to build the compact network
    :return: Compact network
    """
    for module_name, module in pruned.named_modules():
        if len(list(module.children())) == 0:
            new_module = None
            sub_module_names = module_name.split(".")
            dict_name = "{}.weight".format(module_name)
            if isinstance(module, nn.Conv2d):
                new_module = nn.Conv2d(
                    compact_state_dict[dict_name].shape[1],
                    compact_state_dict[dict_name].shape[0],
                    getattr(module, "kernel_size"),
                    getattr(module, "stride"),
                    getattr(module, "padding"),
                    getattr(module, "dilation"),
                    getattr(module, "groups"),
                    getattr(module, "bias") is not None,
                    getattr(module, "padding_mode"),
                )
            elif isinstance(module, nn.Linear):
                new_module = nn.Linear(
                    compact_state_dict[dict_name].shape[1],
                    compact_state_dict[dict_name].shape[0],
                    getattr(module, "bias") is not None
                )
            elif isinstance(module, nn.BatchNorm1d):
                new_module = nn.BatchNorm1d(
                    compact_state_dict[dict_name].shape[0],
                    getattr(module, "eps"),
                    getattr(module, "momentum"),
                    getattr(module, "affine"),
                    getattr(module, "track_running_stats")
                )
            elif isinstance(module, nn.BatchNorm2d):
                new_module = nn.BatchNorm2d(
                    compact_state_dict[dict_name].shape[0],
                    getattr(module, "eps"),
                    getattr(module, "momentum"),
                    getattr(module, "affine"),
                    getattr(module, "track_running_stats")
                )

            substitute_module(pruned, new_module, sub_module_names)

    pruned.load_state_dict(compact_state_dict)

    return pruned
