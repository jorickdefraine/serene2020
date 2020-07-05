import torch
from torch import nn

from .. import utilities


class NeuronLOBSTER:
    def __init__(self, model, lmbda, layers):
        """
        Initialize the LOBSTER regularizer.
        :param model: PyTorch model.
        :param lmbda: Lambda hyperparameter.
        :param layers: Tuple of layer on which apply the regularization e.g. (nn.modules.Conv2d, nn.modules.Linear)
        """
        self.model = model
        self.lmbda = lmbda
        self.layers = layers
        # Dict containing the preactivation of each layer as a Tensor of the layer's dimensions
        self.preactivations = {}

        # Attach to each layer a backward hook that allow us to automatically extract the preactivation
        # during the loss' backward pass
        for n, mo in model.named_modules():
            if len(list(mo.children())) == 0 and len(list(mo.parameters())) != 0:
                print("Attached hook to {}".format(n))
                mo.register_backward_hook(utilities.get_activation(self.preactivations, n, "backward"))

    @torch.no_grad()
    def step(self, masks):
        """
        Regularization step.
        :param masks: Dictionary of type `layer: tensor` containing, for each layer of the network a tensor
        with the same size of the layer that is element-wise multiplied to the layer.
        See `utilities.get_mask_neur` or `utilities.get_mask_par` for an example of mask construction.
        """
        for n_m, mo in self.model.named_modules():
            if isinstance(mo, self.layers):
                for n_p, p in mo.named_parameters():
                    name = "{}.{}".format(n_m, n_p)
                    reshaped = False

                    if "weight" in n_p:

                        # Sensitivity-based weight optimization
                        if len(p.shape) > 2:
                            original_shape = p.shape
                            target_shape = torch.Size([p.shape[0], -1])
                            p = p.view(target_shape)
                            reshaped = True

                        # Compute insensitivity only for weight params
                        # in order to avoid multiple computation of the same value
                        # Tensor [ch out, examples, neur for ch, neur for ch]
                        self.preactivations[n_m] = torch.transpose(self.preactivations[n_m], 0, 1).contiguous()

                        if len(self.preactivations[n_m].shape) > 2:
                            # Tensor [ch out, -1]
                            self.preactivations[n_m] = self.preactivations[n_m].view(self.preactivations[n_m].shape[0],
                                                                                     -1)

                        sensitivity = torch.mean(torch.abs(self.preactivations[n_m]), dim=1)  # |dl/dP|

                        insensitivity = torch.nn.functional.relu(1 - sensitivity)

                        if isinstance(mo, (nn.modules.Conv2d, nn.modules.Linear)):
                            regu = torch.einsum(
                                'ij,i->ij',
                                p,
                                insensitivity
                            )  # neuron-by-neuron (channel-by-channel) w * Ins
                        else:
                            regu = p.mul(insensitivity)

                    else:
                        regu = p.mul(insensitivity)

                    p.add_(-self.lmbda, regu)  # w - lmbd * w * Ins

                    if reshaped:
                        p = p.view(original_shape)

                    utilities.apply_masks(p, masks, name)
