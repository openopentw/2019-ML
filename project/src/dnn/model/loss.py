import torch
import torch.nn.functional as F


def wmse_loss(output, target):
    raw_mse = F.mse_loss(output, target, reduction='none')
    weights = torch.autograd.Variable(torch.Tensor([2 ** 2, 600 ** 2, 3 ** 2])).cuda()
    return (raw_mse * weights).mean()


def nse_loss(output, target):
    raw_mse = F.mse_loss(output, target, reduction='none')
    return (raw_mse / target).mean()
