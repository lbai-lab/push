import torch

# after squeeze:
# prediction of force: [nA*batch_size, 3]
# label of force: [batch_size, nA, 3]


def EnergyLoss(pred, label):
    p, l = pred["E"].squeeze(), label["E"].squeeze()
    mae = torch.nn.L1Loss()
    return mae(p, l)


def AtomForceLoss(pred, label):
    p, l = pred["F"].squeeze(), label["F"].squeeze()
    p = p.reshape((-1, 3))
    l = l.reshape((-1, 3))
    mae = torch.nn.L1Loss()
    f = mae(p, l)
    return f


def EnergyForceLoss(pred, label):
    E = EnergyLoss(pred, label)
    F = AtomForceLoss(pred, label)
    return E + (30)*F
