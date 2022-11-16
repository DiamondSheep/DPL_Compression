import torch
import torch.nn as nn
import torch.nn.functional as F

from operator import attrgetter

def reshape_weight(weight, transpose=True):
    """
    C_out x C_in x k x k -> (C_in x k x k) x C_out.
    """
    if transpose:
        if len(weight.size()) == 4:
            C_out, C_in, k, k = weight.size()
            return weight.view(C_out, C_in * k * k).t()
        else:
            return weight.t()
    else:
        if len(weight.size()) == 4:
            C_out, C_in, k, k = weight.size()
            return weight.view(C_out, C_in * k * k)
        else:
            return weight

def reshape_back_weight(weight, k=3, conv=True, transpose=True):
    """
    (C_in x k x k) x C_out -> C_out x C_in x k x k.
    """
    if transpose:
        if conv:
            C_in_, C_out = weight.size()
            C_in = C_in_ // (k * k)
            return weight.t().view(C_out, C_in, k, k)
        else:
            return weight.t()
    else:
        if conv:
            weight = weight.t()
            C_in_, C_out = weight.size()
            C_in = C_in_ // (k * k)
            return weight.t().view(C_out, C_in, k, k)
        else:
            return weight


def reshape_activations(activations, k=3, stride=(1, 1), padding=(1, 1), groups=1):
    """
    N x C_in x H x W -> (N x H x W) x C_in.
    """

    if len(activations.size()) == 4:
        # gather activations
        a_padded = F.pad(activations, (padding[1], padding[1], padding[0], padding[0]))
        N, C, H, W = a_padded.size()
        a_stacked = []

        for i in range(0, H - k + 1, stride[0]):
            for j in range(0, W - k + 1, stride[1]):
                a_stacked.append(a_padded[:, :, i:i + k, j:j + k])

        # reshape according to weight
        a_reshaped = reshape_weight(torch.cat(a_stacked, dim=0)).t()

        # group convolutions (e.g. depthwise convolutions)
        a_reshaped_groups = torch.cat(a_reshaped.chunk(groups, dim=1), dim=0)

        return a_reshaped_groups

    else:
        return activations

class ActivationWatcher:
    """
    Monitors and stores *input* activations in all the layers of the network.

    Args:
        - model: the model to monitor, should be `nn.module`
        - n_activations: number of activations to store
        - layer: if None, monitors all layers except BN, otherwise single
          layers to monitor

    Remarks:
        - Do NOT use layers with inplace operations, otherwise
          the activations will not be monitored correctly
        - Memory to store activations is pre-allocated for efficiency
    """

    def __init__(self, model, layer=''):
        self.model = model
        # layers to monitor
        all_layers = self._get_layers()
        if len(layer) == 0:
            self.layers = all_layers
        else:
            assert layer in all_layers
            self.layers = [layer]
        # initialization
        self.modules_to_layers = {attrgetter(layer)(self.model): layer for layer in self.layers}
        self._register_hooks()
        self._watch = False

    def _get_layers(self):
        # get proper layer names(without bias)
        keys = self.model.state_dict().keys()
        layers = [k[:k.rfind(".")] for k in keys if 'bias' not in k]
        # remove BN layers
        layers = [layer for layer in layers if not isinstance(attrgetter(layer)(self.model), nn.BatchNorm2d)]

        return layers

    def _get_bn_layers(self):
        # get proper layer names
        keys = self.model.state_dict().keys()
        layers = [k[:k.rfind(".")] for k in keys if 'weight' in k]
        # only keep BN layers
        layers = [layer for layer in layers if isinstance(attrgetter(layer)(self.model), nn.BatchNorm2d)]

        return layers

    def _get_bias_layers(self):
        # get proper layer names
        keys = self.model.state_dict().keys()
        layers = [k[:k.rfind(".")] for k in keys if 'bias' in k]
        # only keep BN layers
        layers = [layer for layer in layers if not isinstance(attrgetter(layer)(self.model), nn.BatchNorm2d)]

        return layers

    def _register_hooks(self):
        # define hook to save output after each layer
        def fwd_hook(module, input, output):
            layer = self.modules_to_layers[module]
            if self._watch:
                # retrieve activations
                activations = input[0].data.cpu()
                # store activations
                self.activations[layer].append(activations)
        # register hooks
        self.handles = []
        for layer in self.layers:
            handle = attrgetter(layer)(self.model).register_forward_hook(fwd_hook)
            self.handles.append(handle)

    def watch(self, loader, criterion, n_iter):
        # watch
        self._watch = True
        # initialize activations storage
        self.activations = {layer: [] for layer in self.layers}
        # gather activations
        evaluate(loader, self.model, criterion, n_iter=n_iter)
        # unwatch
        self._watch = False
        # treat activations
        self.activations = {k: torch.cat(v, dim=0) for (k, v) in self.activations.items()}
        # remove hooks from model
        for handle in self.handles:
            handle.remove()
        # return activations
        return self.activations

    def save(self, path):
        torch.save(self.activations, path)

def compute_size(model):
    """
    Size of model (in MB).
    """

    res = 0
    for n, p in model.named_parameters():
        res += p.numel()

    return res * 4 / 1024 / 1024