import torch
from progress.bar import Bar

@torch.no_grad()
def evaluate(net, val_loader):
    net.eval()
    top5 = AverageMeter()
    top1 = AverageMeter()
    with Bar('Evaluating', fill='=', max=len(val_loader)) as bar:
        for i, data in enumerate(val_loader):
            if torch.cuda.is_available():
                input, label = data[0].cuda(), data[1].cuda()
            else:
                input, label = data[0], data[1]
            output = net(input)
            prec1, prec5 = accuracy(output.data, label.data, topk=(1, 5))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            bar.suffix = f'({i + 1}/{len(val_loader)}) | ETA: {bar.eta_td} | Top1: {top1.avg:.3f}%% | Top5: {top5.avg:.3f}%%'
            bar.next()
    return top1.avg, top5.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    pass