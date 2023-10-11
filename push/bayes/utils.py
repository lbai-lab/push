import torch


def flatten(lst):
    """Inspired by: https://github.com/wjmaddox/swa_gaussian/blob/master/swag/utils.py"""
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
    """ Inspired by: https://github.com/wjmaddox/swa_gaussian/blob/master/swag/utils.py"""
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    return outList
