from torch import nn

# ArgumentParser configs
AVAILABLE_MODELS = ["lenet300", "lenet5", "alexnet", "resnet32", "resnet18", "resnet50", "resnet101", "lenet5_8_16", "lenet5_16_32", "lenet300_32"]
AVAILABLE_DATASETS = ["mnist", "fashion-mnist", "cifar10", "cifar100", "imagenet", "cockpit18"]
AVAILABLE_SENSITIVITIES = ["lobster", "neuron-lobster", "serene"]
AVAILABLE_SERENE = ["full", "lower-bound", "local"]

# Logs root directory
LOGS_ROOT = "logs"

# Layers considered during regularization and pruning
LAYERS = (nn.modules.Linear, nn.modules.Conv2d)
