import torch


def wmae_metric(output, target):
    with torch.no_grad():
        assert output.shape[0] == len(target)
        weights = torch.autograd.Variable(torch.Tensor([2, 600, 3])).cuda()
    return (torch.abs(output - target) * weights).sum(1).mean()


def nae_metric(output, target):
    with torch.no_grad():
        assert output.shape[0] == len(target)
    return (torch.abs(output - target) / target).sum(1).mean()
