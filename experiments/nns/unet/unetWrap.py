
import torch.nn as nn
from nns.unet.unet import UNet1d


class UNet1dWrap(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn = UNet1d(in_channels, out_channels)

    def forward(self, x):
        return self.nn.forward(x).permute([0, 2, 1])


def unet_loss_fn(x, y):
    _batch = y.size(0)
    loss = nn.MSELoss()(x.reshape(_batch, -1), y.reshape(_batch, -1))
    return loss


def unet_loss_fn2(x, y):
    _batch = y.size(0)
    loss = nn.MSELoss()(x.reshape(_batch, -1), y.reshape(_batch, -1))
    return loss
