import torch
import argparse
from torch.utils.data import DataLoader
# from .util import * 

# after squeeze:
# prediction of force: [nA*batch_size, 3]
# label of force: [batch_size, nA, 3]

# Total Loss ########################################################################################################
def Fz_F2_ELoss(pred, label, args):
    if args.cur_epoch <= args.freeze_epoch:
        return AtomForce2Loss(pred, label, args)
    else:   
        return EnergyLoss(pred, label, args)

def Fz_F_ELoss(pred, label, args):
    if args.cur_epoch <= args.freeze_epoch:
        return AtomForceLoss(pred, label, args)
    else:   
        return EnergyLoss(pred, label, args)
        
def Dir_EFLoss(pred, label, args):
    if args.cur_epoch < args.epoch//2:
        return DirectionLoss(pred, label, args)
    else:   
        E = EnergyLoss(pred, label, args)
        F = AtomForceLoss(pred, label, args)
        return E + 30*F
    
def F_ELoss(pred, label, args):
    if args.cur_epoch < args.epoch//2:
        return AtomForceLoss(pred, label, args)
    else:   
        return EnergyLoss(pred, label, args)

def F50_E10Loss(pred, label, args):
    if args.cur_epoch%60 < 50:
        return AtomForceLoss(pred, label, args)
    else:   
        return EnergyLoss(pred, label, args)

def F10_E10Loss(pred, label, args):
    if args.cur_epoch%20 < 10:
        return AtomForceLoss(pred, label, args)
    else:   
        return EnergyLoss(pred, label, args)

def E_FLoss(pred, label, args):
    if args.cur_epoch < args.epoch//2:
        return EnergyLoss(pred, label, args)
    else:   
        return AtomForceLoss(pred, label, args)
        

def EnergyForceLoss(pred, label):
    E = EnergyLoss(pred, label)
    F = AtomForceLoss(pred, label)
    return E + (30)*F
    #return E + (30)*F, E, F

def EnergyAxisCosForceLoss(pred, label, args):
    E = EnergyLoss(pred, label, args)
    F = AxisCosForceLoss(pred, label, args)
    return E + 30*F

def EnergyDirForceLoss(pred, label, args):
    E = EnergyLoss(pred, label, args)
    F = DirectionLoss(pred, label, args)
    return E + 30*F

def EnergyDirAtomForceLoss(pred, label, args):
    E = EnergyLoss(pred, label, args)
    F1 = DirectionLoss(pred, label, args)
    F2 = AtomForceLoss(pred, label, args)
    return E + 50*F1 + 30*F2

def FullCovLoss(pred, label, args):
    ep, el = pred["E"], label["E"]
    fp, fl = pred["F"].squeeze(), label["F"].squeeze()
    fp = fp.reshape((ep.shape[0], -1))
    fl = fl.reshape((ep.shape[0], -1))
    diff = torch.cat((ep-el, fp-fl), axis=1)
    covariance_inv = label["covariance_inv"]

    return torch.mean(0.5*(diff @ covariance_inv @ (diff.t()))) 

def SubCovLoss(pred, label, args):
    ep, el = pred["E"], label["E"]
    fp, fl = pred["F"].squeeze(), label["F"].squeeze()
    fp = fp.reshape((ep.shape[0], -1))
    fl = fl.reshape((ep.shape[0], -1))
    covariance_inv = label["covariance_inv"]
    mean = label["mean"]

    p = torch.cat((ep, fp), axis=1)
    p = p-mean
    l = torch.cat((el, fl), axis=1)
    l = l-mean
    mae = torch.nn.L1Loss()

    return mae((p @ covariance_inv @ (p.t())), (l @ covariance_inv @ (l.t()))  )

def SubCovLoss0(pred, label, args):
    ep, el = pred["E"], label["E"]
    fp, fl = pred["F"].squeeze(), label["F"].squeeze()
    fp = fp.reshape((ep.shape[0], -1))
    fl = fl.reshape((ep.shape[0], -1))

    p = torch.cat((ep, fp), axis=1)
    l = torch.cat((el, fl), axis=1)
    mae = torch.nn.L1Loss()

    return mae(torch.matmul(p, p.t()), torch.matmul(l, l.t()))   

def SynthMseLoss(pred, label, args):
    return Y_MseLoss(pred, label, args) + 30*dY_MseLoss(pred, label, args)

# Enery Loss ########################################################################################################
def EnergyLoss(pred, label):
    p, l = pred["E"].squeeze(), label["E"].squeeze()
    mae = torch.nn.L1Loss()
    return mae(p, l)

# Force Loss ########################################################################################################
def PosForceLoss(pred, label):
    # not for training, for predicting
    p, l = pred["F"], label["F"]
    batch_size = l.shape[0]   
    p = p.reshape(batch_size, -1).t()
    l = l.reshape(batch_size, -1).t()
    #print("p:", p[0])
    #print("l:", l[0])
    mae = torch.nn.L1Loss()
    loss = [mae(p[i], l[i]) for i in range(len(p))]
    #print([t.item() for t in loss[:10]])
    return loss

def AxisCosForceLoss(pred, label, args):
    p, l = pred["F"].squeeze(), label["F"].squeeze()
    p = p.reshape((-1, 1))
    l = l.reshape((-1, 1))
    CosS = (p*l+1)/((torch.sqrt(p*p+1))*(torch.sqrt(l*l+1)))
    return sum(torch.exp(CosS))

def AtomForceLoss(pred, label):
    p, l = pred["F"].squeeze(), label["F"].squeeze()
    p = p.reshape((-1, 3))
    l = l.reshape((-1, 3))
    mae = torch.nn.L1Loss()
    f = mae(p, l)
    return f

def AtomForce2Loss(pred, label, args):
    p, l = pred["F"].squeeze(), label["F"].squeeze()
    p = p.reshape((-1, 3))
    l = l.reshape((-1, 3))
    mse = torch.nn.MSELoss()
    f = mse(p, l)
    return f

def DirectionLoss(pred, label, args):
    p, l = pred["F"].squeeze(), label["F"].squeeze()
    p = p.reshape((-1, 1))
    l = l.reshape((-1, 1))
    diff = torch.abs(p-l)

    positive = torch.zeros_like(p)
    negative = diff
    loss = torch.where(p*l>=0, positive, negative)

    return torch.sum(loss)

def WeightednegLoss(pred, label, args):
    p, l = pred["F"].squeeze(), label["F"].squeeze()
    p = p.reshape((-1, 1))
    l = l.reshape((-1, 1))
    diff = torch.abs(p-l)

    positive = diff
    negative = diff*30
    loss = torch.where(p*l>=0, positive, negative)

    return torch.sum(loss)

def DirectionExpLoss(pred, label, args):
    p, l = pred["F"].squeeze(), label["F"].squeeze()
    p = p.reshape((-1, 1))
    l = l.reshape((-1, 1))
    diff = torch.abs(p-l)
    diff = torch.nn.functional.normalize(diff)

    positive = torch.zeros_like(p)
    negative = torch.exp(diff)
    loss = torch.where(p*l>=0, positive, negative)

    return torch.sum(loss)

def DirectionDiffExpLoss(pred, label, args):
    p, l = pred["F"].squeeze(), label["F"].squeeze()
    p = p.reshape((-1, 1))
    l = l.reshape((-1, 1))
    diff = torch.abs(p-l)
    diff = torch.nn.functional.normalize(diff)

    positive = diff
    negative = torch.exp(diff)
    loss = torch.where(p*l>=0, positive, negative)

    return torch.sum(loss)

# Y value loss
def Y_MseLoss(pred, label, args):
    p = pred[:,0]
    l = label[:,0]
    mse = torch.nn.MSELoss() 
    return mse(p, l)

# dY value loss
def dY_MseLoss(pred, label, args):
    p = pred[:,1:]
    l = label[:,1:]
    mse = torch.nn.MSELoss() 
    return mse(p, l)

# PINN loss
def Loss_SDGR(pred, label):
    mse = torch.nn.MSELoss()

    u0 = mse(label["u0"], pred["u0"])
    v0 = mse(label["v0"], pred["v0"])
    ub = mse(pred["u_lb"], pred["u_ub"])
    vb = mse(pred["v_lb"], pred["v_ub"])
    uxb = mse(pred["u_x_lb"], pred["u_x_ub"])
    vxb = mse(pred["v_x_lb"], pred["v_x_ub"])
    fu = torch.mean(pred["f_u"]*pred["f_u"]) 
    fv = torch.mean(pred["f_v"]*pred["f_v"])
    total = u0 + v0 + ub + vb + uxb + vxb + fu + fv 

    loss_dict = {"u0": u0.item(), "v0": v0.item(), 
                 "ub": ub.item(), "vb": vb.item(),  
                 "uxb": uxb.item(), "vxb": vxb.item(),  
                 "fu": fu.item(), "fv": fv.item(),
                 "total": total.item()}

    return total, loss_dict

def Loss_NS(pred, label):
    mse = torch.nn.MSELoss()

    u = mse(label["u"], pred["u"])
    v = mse(label["v"], pred["v"]) 
    fu = torch.mean(pred["f_u"]*pred["f_u"]) 
    fv = torch.mean(pred["f_v"]*pred["f_v"])
    total = u + v + fu + fv

    loss_dict = {"u": u.item(), "v": v.item(), "fu": fu.item(), "fv": fv.item(),
                 "total": total.item()}

    return total, loss_dict

def Loss_KDV(pred, label):
    mse = torch.nn.MSELoss()

    u0 = mse(label["u0"], pred["u0"]) 
    u1 = mse(label["u1"], pred["u1"]) 
    total = u0 + u1

    loss_dict = {"u0": u0.item(), "u1": u1.item(),
                 "total": total.item()}
    
    return total, loss_dict

def Loss_AC(pred, label):
    mse = torch.nn.MSELoss()
    u0 = mse(label["u0"], pred["u0"])
    u1 = mse(pred["u1"][0,:], pred["u1"][1,:])
    u1x = mse(pred["u1_x"][0,:], pred["u1_x"][1,:])
    total = u0 + u1 + u1x 

    loss_dict = {"u0": u0.item(), "u1": u1.item(), "u1x":u1x.item(),
                 "total": total.item()}
    
    return total, loss_dict

