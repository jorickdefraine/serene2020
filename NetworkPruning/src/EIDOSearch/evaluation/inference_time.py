import sys
import time

import numpy as np
import torch


@torch.no_grad()
def inference_time(model, data_dim, device="cpu", evaluation_iterations=1, num_threads=1):
    """
    Measure inference time for a given model and input size.
    :param model: PyTorch model.
    :param data_dim: Dimension of the input to feed to the model as tuple e.g. (1, 1, 28, 28) for MNIST.
    :param device: Device on which run the inference, cpu or indexed cuda device.
    :param evaluation_iterations: How many forward pass to perform.
    :param num_threads: Number of CPU threads to use.
    :return: Inference time averaged over evaluation_iterations forward passes and its standard deviation.
    """
    if device != "cpu" and "cuda" not in device:
        raise ValueError("Device must be \'cpu\' or \'cuda\', given \'{}\'".format(device))

    if "cuda" in device:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model.to(device)
    model.eval()

    elapsed = []
    torch.set_num_threads(num_threads)

    for i in range(evaluation_iterations):
        data = torch.randn(data_dim)

        if "cuda" in device:
            data = data.to(device)
            torch.cuda.synchronize(torch.device(device))

        start = time.perf_counter() if sys.version_info >= (3, 3) else time.clock()

        model(data)

        if "cuda" in device:
            torch.cuda.synchronize(torch.device(device))

        elapsed.append(time.perf_counter() - start)

    print(np.max(elapsed), np.min(elapsed))

    return np.mean(elapsed), np.std(elapsed)
