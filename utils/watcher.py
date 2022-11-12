import torch
import torch.nn as nn

from operator import attrgetter

def evaluate(test_loader, model, criterion, n_iter=-1, verbose=False, device='cuda'):
    """
    Standard evaluation loop.
    """

    n_iter = len(test_loader) if n_iter == -1 else n_iter
    modulo = 0 if verbose else -1
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(test_loader), batch_time, losses, top1, top5, prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(test_loader):
            # early stop
            if i >= n_iter: break

            # cuda
            
            input = input.cuda() if device == 'cuda' else input
            target = target.cuda() if device == 'cuda' else target
            
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == modulo:
                progress.print(i)

        return top1.avg

class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """
    Pretty and compact metric printer.
    """

    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

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
