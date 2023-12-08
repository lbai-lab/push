import torch


def flatten(lst):
    """
    Flatten a list of tensors into a 1D tensor. Inspired by: https://github.com/wjmaddox/swa_gaussian/blob/master/swag/utils.py

    Args:
        lst (list): List of tensors to be flattened.

    Returns:
        torch.Tensor: Flattened 1D tensor.

    """
    """"""
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
    """
    Unflatten a 1D tensor into a list of tensors shaped like likeTensorList. Inspired by: https://github.com/wjmaddox/swa_gaussian/blob/master/swag/utils.py

    Args:
        vector (torch.Tensor): 1D tensor to be unflattened.
        likeTensorList (list): List of tensors providing the shape for unflattening.

    Returns:
        list: List of unflattened tensors.

    """
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
