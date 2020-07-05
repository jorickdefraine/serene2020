from copy import deepcopy
from math import inf

from ...pruning import *


def compact_network(pruned_model, pinned_in, pinned_out):
    """
    Compact the network state_dict
    :param pruned_model: Sparse model to be compacted
    :return: Compact state_dict
    """
    compact_state_dict = update_bias(pruned_model, pinned_in, pinned_out)
    compact_state_dict = remove_zero_element(compact_state_dict, pruned_model, pinned_in, pinned_out)

    return compact_state_dict


def update_bias(pruned_model, pinned_in, pinned_out):
    """
    Propagate the biases of remove neurons which still have a non-zero bias
    :param pruned_model: Sparse model to be compacted
    :return: state_dict with updated biases
    """
    state = deepcopy(pruned_model.state_dict())
    zeros = []
    non_zeros = []
    biases = None
    was_conv = False
    propagate_bias = False
    prev_module = None

    # Iterate through all the element of the state_dict
    for k in state:
        # print(k)
        # If the element is a weight we use it to eventually update the corresponding biases
        if "weight" in k:
            if k.replace(".weight", "") in pinned_out:
                continue

            # Find in the pruned model the layer that corresponds to the current state_dict element
            current_module, next_module = find_module(pruned_model, k)

            if current_module is None:
                raise RuntimeError('The supplied model does not contain a module corresponding to key {}'.format(k))

            # Select the activation function
            if isinstance(prev_module, (nn.ReLU, nn.ReLU6, nn.LeakyReLU)):
                activation_function = prev_module
            else:
                activation_function = nn.Identity()

            bias_key = k.replace("weight", "bias")

            # The current element is a convolutional layer
            if isinstance(current_module, nn.Conv2d):
                # Sum the convolutional values for dimensions: input, h, w
                conv_sum = torch.sum(torch.abs(state[k]), dim=(1, 2, 3))

                # If the current layer has a bias or the previous layer had a bias we may have to update it
                if bias_key in state or propagate_bias:
                    propagate_bias = False
                    # Memorize that we encountered a conv layer
                    was_conv = True
                    # Define the bias update tensor initialized at zero
                    update = torch.zeros(state[k].shape[0])
                    # Check if we have biases coming from the previous layer
                    if biases is not None and torch.sum(biases) != 0:
                        # Apply the activation function to the biases of the previous layer
                        biases = activation_function(biases)
                        # Take the biases to be propagated one by one
                        for i in range(biases.shape[0]):
                            bias = biases[i]
                            if bias != 0:
                                # Manually execute the forward propagation of the bias in order to obtain it's weighted value
                                # Add each weighted sum to the update value
                                bias_prop = state[k][:, i, :, :].mul(bias)
                                update += torch.sum(torch.sum(bias_prop, dim=2), dim=1)

                        if torch.sum(torch.abs(update)) != 0:
                            if bias_key in state:
                                state[bias_key].add_(update)
                    if bias_key in state:
                        # If the next layer is a convolutional with padding we do not propagate the bias
                        # Get current layer biases
                        biases = (state[bias_key]).clone().detach()
                        # Set to 0 the biases corresponding to non-zero filters, this are not propagated to the next layer
                        biases[torch.where(conv_sum != 0)[0]] = 0
                    # This layer has no natural biases but we have value propagated from the previous layer
                    # The computed update are the "ghost biases" of this layer that will be propagated to the next layer with bises
                    biases = update

                    # Signal that we have biases to propagate to the next layer
                    if biases is not None and torch.sum(biases) != 0:
                        propagate_bias = True

                    # Memorize the number of output channel of the current conv layer
                    out_ch_num = state[k].shape[0]

                # Identify the zeroed and the non-zero filters index
                zeros = (conv_sum == 0).nonzero()
                non_zeros = (conv_sum != 0).nonzero()

            # The current element is a linear layer
            if isinstance(current_module, nn.Linear):
                # If the current layer has a bias or the previous layer had a bias we may have to update it
                if bias_key in state or propagate_bias:
                    propagate_bias = False
                    if biases is not None and torch.sum(biases) != 0:
                        # Apply the activation function to the bieses of the previous layer
                        biases = activation_function(biases)

                        # The previous layer was a convolutional
                        if was_conv:
                            was_conv = False
                            # Evaluate how many FC neurons correspond to the previous conv out channel
                            # current_layer_neurons_number / previous_layer_output_channels
                            neurons_per_channel = int(state[k].shape[1] / out_ch_num)
                            # Define the bias update tensor initialized at zero
                            update = torch.zeros(state[k].shape[0])

                            # Take the zeroed filters index one by one
                            for z in zeros:
                                # Compute the starting and end index of the neurons that correspond to the filter
                                from_idx = z * neurons_per_channel
                                to_idx = (z + 1) * neurons_per_channel
                                # Get the bias corresponding to the filter
                                bias = biases[z]
                                # Manually compute the weighted sum between the bias and the weights of the neurons in the previously defined range
                                # Add each sum to the update value
                                update += torch.sum(state[k][:, from_idx:to_idx].mul(bias), dim=1)

                        # The previous layer was a linear
                        else:
                            # Multiply the previous layer biases by the current element weights
                            biases = biases * state[k]
                            # Define the update value for the biases, summing the values related to the same neuron
                            update = torch.sum(biases, dim=1)

                        # Update the biases of the current layer
                        if torch.sum(torch.abs(update)) != 0:
                            if bias_key in state:
                                state[bias_key].add_(update)

                    if bias_key in state:
                        # Get current layer biases
                        biases = (state[bias_key]).clone().detach()

                        # Set to 0 the biases corresponding to non-zero neurons, this are not propagated to the next layer
                        if torch.sum(torch.abs(biases)) != 0:
                            for col in range(state[k].shape[0]):
                                if torch.sum(torch.abs(state[k][col])) != 0:
                                    biases[col] = 0
                    # This layer has no natural biases but we have value propagated from the previous layer
                    # The computed update are the "ghost biases" of this layer that will be propagated to the next layer with bises
                    if bias_key not in state:
                        biases = update

                    # Signal that we have biases to propagate to the next layer
                    if torch.sum(torch.abs(biases)) != 0:
                        propagate_bias = True

            prev_module = next_module

    return state


def remove_zero_element(state, pruned_model, pinned_in, pinned_out):
    """
    Compact the state_dict removing all the zeroed neurons, corresponding connections and biases
    :param pruned_model: Sparse model to be compacted
    :param state: state_dict to be compacted
    :return: Compact state_dict
    """
    zeros = []
    non_zeros = []
    was_conv = False
    first = True

    # Iterate through all the elements of the state_dict
    for k in state:
        # print(k)
        if "weight" in k:

            # Find in the pruned model the layer that corresponds to the current state_dict element
            current_module, next_module = find_module(pruned_model, k)

            if current_module is None:
                raise RuntimeError('The supplied model does not contain a module corresponding to key {}'.format(k))

            bias_key = k.replace("weight", "bias")

            # The current element is a convolutional layer
            if isinstance(current_module, nn.Conv2d):
                # If the next layer is a convolutional with padding we do not remove the current layer neurons
                # Memorize that we encountered a conv layer
                was_conv = True
                # Sum the convolutional values for dimensions: input, h, w
                conv_sum = torch.sum(torch.abs(state[k]), dim=(1, 2, 3))

                # TODO add comment
                if first:
                    if k.replace(".weight", "") not in pinned_in:
                        stay_idx_prev = torch.where(torch.sum(torch.abs(state[k]), dim=(0, 2, 3)) != 0)[0]
                    else:
                        stay_idx_prev = torch.where(torch.sum(torch.abs(state[k]), dim=(0, 2, 3)) >= 0)[0]
                else:
                    stay_idx_prev = non_zeros.view(-1)

                # Get which filters of the current layer are zeroed i.e. the sum of its element in ABS() must be = 0
                zeros = (conv_sum == 0).nonzero() if k.replace(".weight", "") not in pinned_out else (
                        conv_sum < 0).nonzero()
                # Get which filters of the current layer are NON zero i.e. the sum of its element in ABS() must be != 0
                non_zeros = (conv_sum != 0).nonzero() if k.replace(".weight", "") not in pinned_out else (
                        conv_sum >= 0).nonzero()
                # Get the number of output channels
                out_ch_num = state[k].shape[0]

                # Remove from the current layer all the zeroed filters
                stay_idx = torch.where(conv_sum != 0)[0] if k.replace(".weight", "") not in pinned_out \
                    else torch.where(conv_sum >= 0)[0]
                remove_idx = torch.where(conv_sum == 0)[0] if k.replace(".weight", "") not in pinned_out \
                    else torch.where(conv_sum < 0)[0]

                # IN ch
                state[k] = state[k][:, stay_idx_prev, :, :] if k.replace(".weight", "") not in pinned_in else state[k]
                # OUT ch
                state[k] = state[k][stay_idx, :, :, :]

                # Set to inf the biases corresponding to zeroed filters in the actual state_dict, marking them as "to remove"
                if bias_key in state:
                    state[bias_key][remove_idx] = inf

            # The current element is a linear layer
            if isinstance(current_module, nn.Linear):
                if bias_key in state:
                    for row in range(state[k].shape[0]):
                        # Set to inf the biases corresponding to zeroed neurons in the actual state_dict, marking them as "to remove"
                        if torch.sum(torch.abs(state[k][row])) == 0:
                            state[bias_key][row] = inf

                # The previous layer was a convolutional
                if was_conv:
                    was_conv = False
                    # Evaluate how many FC neurons correspond to the previous CONV out channel
                    neurons_per_channel = int(state[k].shape[1] / out_ch_num)
                    remaining_neurons = []
                    for z in non_zeros:
                        # Compute the starting and end index of such neurons
                        from_idx = z * neurons_per_channel
                        to_idx = (z + 1) * neurons_per_channel
                        remaining_neurons.append(state[k][:, from_idx:to_idx])

                # FC layer after a FC layer
                else:
                    # Set to zero all the connection of the current layer corresponding to zeroed neurons of the previous layer
                    remaining_neurons = []
                    for z in non_zeros:
                        remaining_neurons.append(state[k][:, z])

                # Get which neurons of the current channel are zeroed
                zeros = (torch.sum(torch.abs(state[k]), dim=1) == 0).nonzero()
                non_zeros = (torch.sum(torch.abs(state[k]), dim=1) != 0).nonzero()

                # Remove from the current layer all zeroed neurons and connections
                state[k] = torch.cat(remaining_neurons, 1) if remaining_neurons else state[k]
                state[k] = state[k][non_zeros.view(-1)]

        # Remove inf biases
        if "bias" in k:
            state[k] = state[k][state[k] != inf]

        first = False

    return state
