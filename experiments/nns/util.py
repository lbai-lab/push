import csv
import glob
import torch
import argparse
# from simple_term_menu import TerminalMenu

# import loss_fn
from .losses import *

# import models
from nns.schnet.schnet import SchNetWrap
# from models.nn.schnet_bias.schnet_bias import SchNetWrap_bias
# from nns.cgcnn.cgcnn import CGCNN

def get_model(model):
    # Quantum Chemistry
    if model == "schnet": return SchNetWrap(regress_forces=True)
    # if args.model == "schnet_bias": return SchNetWrap_bias(args, regress_forces=args.force)
    if model == "cgcnn": return CGCNN(regress_forces=True)
    # if args.model == "cgcnn_bias": return CGCNN_bias(regress_forces=args.force)
    # if args.model == "dimenet2": return DimeNetPlusPlusWrap(regress_forces=args.force)
    # if args.model == "dimenet2_bias": return DimeNetPlusPlusWrap_bias(regress_forces=args.force)
    # if args.model == "gemnet": return GemNetT(regress_forces=args.force)
    # if args.model == "gemnet_bias": return GemNetT_bias(regress_forces=args.force)
    # if args.model == "forcenet": return ForceNet(regress_forces=args.force)
    # if args.model == "forcenet_bias": return ForceNet_bias(regress_forces=args.force)
    # if args.model == "trans_encoder": return trans_encoder(args)

    # # Synthetic function
    # if args.model == "synth_mlp": return Synth_mlp(args.dim)
    # if args.model == "synth_trans": return Synth_trans(args.dim)
    # if args.model == "synth_rnn": return Synth_rnn(args.dim)

    # PINNs
    backbone = get_backbone(args)
    backbone.lb = backbone.lb#.to(args.device)
    backbone.ub = backbone.ub#.to(args.device)

    if args.model == "SDGR": return SDGR(backbone)
    if args.model == "NS": return NS(backbone)
    if args.model == "KDV": return KDV(backbone, args.dt)
    if args.model == "AC": return AC(backbone, args.dt)

    print("model name incorrect")
    return None

# def get_data(args):
#     if args.model == "SDGR": return prep_data_SDGR(args)
#     if args.model == "NS":   return prep_data_NS(args)
#     if args.model == "KDV":  return prep_data_KDV(args)
#     if args.model == "AC":   return prep_data_AC(args)

#     print("pinn name incorrect")
#     return None, None, None

def get_dataloader(args):
    if args.model == "SDGR":
        dataset = SDGRDataset(path_to_dataset="datasets/pinn/")
        args.lb=dataset.lb
        args.ub=dataset.ub

    elif args.model == "NS":
        dataset = NSDataset(path_to_dataset="datasets/pinn/")
        args.lb=dataset.lb
        args.ub=dataset.ub

    elif args.model == "KDV":
        dataset = KDVDataset(path_to_dataset="datasets/pinn/")
        args.dt=dataset.dt
        args.lb=dataset.lb
        args.ub=dataset.ub

    elif args.model == "AC":
        dataset = ACDataset(path_to_dataset="datasets/pinn/")
        args.dt=dataset.dt
        args.lb=dataset.lb
        args.ub=dataset.ub
    else:
        raise ValueError("Invalid dataset type")
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate)
    return dataloader

# def get_dataset(args):
#     # Quantum Chemistry
#     if args.dataset == "MD17SingleDataset": return MD17SingleDataset(args.style, args.molecule, args.task, args.split, root=args.root)
#     if args.dataset == "MD17Dataset": return MD17Dataset(args.style, args.task, args.split, root=args.root)

#     # Synthetic function
#     if args.dataset == "Branin": return SyntheticDataset(args.root, args.dataset, args.num, args.task)

#     print("dataset name incorrect")
#     return None

def get_loss_fn(args):
    # Quantum Chemistry
    if args.loss_fn == "EnergyLoss": return EnergyLoss
    
    if args.loss_fn == "AxisCosForceLoss": return AxisCosForceLoss
    if args.loss_fn == "DirectionLoss": return DirectionLoss
    if args.loss_fn == "DirectionExpLoss": return DirectionExpLoss
    if args.loss_fn == "DirectionDiffExpLoss": return DirectionDiffExpLoss
    if args.loss_fn == "AtomForceLoss": return AtomForceLoss
    if args.loss_fn == "WeightednegLoss": return WeightednegLoss

    if args.loss_fn == "Fz_F_ELoss": return Fz_F_ELoss
    if args.loss_fn == "Fz_F2_ELoss": return Fz_F2_ELoss

    if args.loss_fn == "Dir_EFLoss": return Dir_EFLoss
    if args.loss_fn == "F_ELoss": return F_ELoss
    if args.loss_fn == "F50_E10Loss": return F50_E10Loss
    if args.loss_fn == "F10_E10Loss": return F10_E10Loss
    if args.loss_fn == "E_FLoss": return E_FLoss

    if args.loss_fn == "EnergyForceLoss": return EnergyForceLoss
    if args.loss_fn == "EnergyDirForceLoss": return EnergyDirForceLoss
    if args.loss_fn == "EnergyDirAtomForceLoss": return EnergyDirAtomForceLoss
    if args.loss_fn == "EnergyAxisCosForceLoss": return EnergyAxisCosForceLoss
    if args.loss_fn == "FullCovLoss": return FullCovLoss
    if args.loss_fn == "SubCovLoss": return SubCovLoss
    if args.loss_fn == "SubCovLoss0": return SubCovLoss0

    # Synthetic function
    if args.loss_fn == "SynthMseLoss": return SynthMseLoss

    # PINNs
    if args.loss_fn == "SDGR": return Loss_SDGR
    if args.loss_fn == "NS": return Loss_NS
    if args.loss_fn == "KDV": return Loss_KDV
    if args.loss_fn == "AC": return Loss_AC

    print("loss function name incorrect")
    return None

def get_optimizer(model, args):
    if args.optimizer == "Adam": return torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.optimizer == "SGD": return torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)

    print("optimizer name incorrect")
    return None

def get_scheduler(optimizer, scheduler):
    # if scheduler == "StepLR": return torch.optim.lr_scheduler.StepLR(optimizer, step_size=steps_per_epoch * 100000, gamma=0.96)
    if scheduler == "ExponentialLR": return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96, last_epoch=- 1, verbose=False)
    if scheduler == "LambdaLR": return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1**(epoch//10), last_epoch=-1)
    if scheduler == "Cosine": return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0, last_epoch=- 1, verbose=False)

    print("scheduler name incorrect")
    return None

# def get_activation(activation, n):
#     if activation == "int_relu": return relu_int(n)
#     if activation == "int_leaky_relu": return leaky_relu_int(n)
#     if activation == "int_elu": return elu_int(n)

#     if activation == "sig_int_relu": return sig_relu_int(n)  
#     if activation == "sig_int_leaky_relu": return sig_leaky_relu_int(n)  
#     if activation == "sig_int_elu": return sig_elu_int(n)  

#     print("activation name incorrect")
#     return None

# def replace_activation(model, args):
#     act = get_activation(args.activation, args.npow)
#     sig_act = get_activation("sig_"+args.activation, args.npow)

#     if isinstance(model, SchNetWrap):
#         model.interactions[0].mlp[1] = act
#         model.interactions[0].act = act
#         model.interactions[1].mlp[1] = act
#         model.interactions[1].act = act
#         model.interactions[2].mlp[1] = act
#         model.interactions[2].act = act
#         model.interactions[3].mlp[1] = act
#         model.interactions[3].act = act
#         model.interactions[4].mlp[1] = act
#         model.interactions[4].act = act
#         model.act = sig_act
#     elif isinstance(model, CGCNN):
#         model.conv_to_fc[1] = act
#         model.fcs[1] = act
#         model.fcs[3] = sig_act
#     elif isinstance(model, DimeNetPlusPlusWrap):
#         model.emb.act =  sig_act
#         model.output_blocks[0].act = sig_act
#         model.output_blocks[1].act = sig_act
#         model.output_blocks[2].act = sig_act
#         model.output_blocks[3].act = sig_act
#         model.interaction_blocks[0].act = act
#         model.interaction_blocks[1].act = act
#         model.interaction_blocks[2].act = act
#     elif hasattr(model, 'nn'): 
#         if isinstance(model.nn, NN_MLP):
#             model.nn.layers[0][1] = act
#             model.nn.layers[1][1] = act
#             model.nn.layers[2][1] = act
#             model.nn.layers[3][1] = act
#         elif isinstance(model.nn, NN_TRANS):
#             for i in range(len(model.nn.layers)):
#                 model.nn.layers[i][1] = act
#             for i in range(len(model.nn.trans_encoder.layers)):
#                 model.nn.trans_encoder.layers[i].activation = act
#         else:
#             print("pinn activaiton not specified")
#             assert False
#     else:
#         print("model activaiton not specified")
#         assert False

#     return

# def list_model_table(nn = "schnet"):
#     # show menu for predict script to choose checkpoint
#     models = glob.glob(f"../../{nn}/checkpoints/*/")
#     models = sorted(models, reverse=True)

#     options = []
#     for model in models:
#         configs = dict()
#         with open(f"{model}info.txt") as fp:
#             for line in fp:
#                 line = line.strip().split(" ")
#                 configs[line[0]] = line[1]

#             dataset = configs["dataset"]
#             molecule = configs["molecule"]
#             loss_fn = configs["loss_fn"]
#             time = model.split("/")[-2]

#         options.append(f"{time}, {dataset}, {molecule}, {loss_fn}")
    
#     terminal_menu = TerminalMenu(options)
#     idx = terminal_menu.show()
#     return models[idx][:-1]

# def get_pretrain_model(args):
#     # load pretained checkpoint
#     models = glob.glob(f"{args.pre_train}/*.pth")
#     models = sorted(models, key=lambda x: float(x.split("/")[-1][4:-4]))
#     args.start_epoch = 0
#     print(f"Using model {models[0]}")

#     return torch.load(models[0], map_location=torch.device(torch.cuda.current_device()), pickle_module=dill)

# def get_best_model(args):
#     # load a checkpoint with the best loss
#     models = glob.glob(f"{args.model_path}/*.pth")
#     models = sorted(models, key=lambda x: float(x.split("/")[-1][4:-4]))
#     print(f"Using model {models[0]}")

#     return torch.load(models[0], map_location=torch.device(torch.cuda.current_device()), pickle_module=dill)

# def get_latest_model(args):
#     # load last checkpont (aka epoch 299)
#     models = glob.glob(f"{args.cont}/*.pth")
#     models = sorted(models, key=lambda x: int(x.split("/")[-1].split("_")[0]))
#     args.start_epoch = int(models[-1].split("/")[-1].split("_")[0])+1
#     print(f"Using model {models[-1]}")

#     return torch.load(models[-1], map_location=torch.device(torch.cuda.current_device()), pickle_module=dill)

# def save_reult(args, identifier, preds, losses):
#     # save after prediction
#     if preds:
#         csvfile = open(f"{args.model_path}/pred_{identifier}.csv", "w", newline='')
#         csvwriter = csv.writer(csvfile)
#         csvwriter.writerows(preds)

#     if losses:
#         csvfile = open(f"{args.model_path}/loss_{identifier}.csv", "w", newline='')
#         csvwriter = csv.writer(csvfile)
#         csvwriter.writerows(losses)
#     return

def _check_bn_apply(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True

def _check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn_apply(module, flag))
    return flag[0]

def _reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)

def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum

def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]

def bn_update(loader, model, device="cuda"):
        if not _check_bn(model):
            return
        was_training = model.training
        model.train()
        momenta = {}
        model.apply(_reset_bn)
        model.apply(lambda module: _get_momenta(module, momenta))
        n = 0

        for input, _ in loader:
            momentum = 1.0
            for module in momenta.keys():
                module.momentum = momentum

            input = {i:v.to(device) for i, v in input.items()}

            model(input)
            n += b

        model.apply(lambda module: _set_momenta(module, momenta))
        model.train(was_training)

def bn_update_pinn(data, model, device="cuda"):
        if not _check_bn(model):
            return
        was_training = model.training
        model.train()
        momenta = {}
        model.apply(_reset_bn)
        model.apply(lambda module: _get_momenta(module, momenta))
        n = 0

        momentum = 1.0
        for module in momenta.keys():
            module.momentum = momentum

        data = {i:v.to(device) for i, v in data.items()}

        model(data)
        n += b

        model.apply(lambda module: _set_momenta(module, momenta))
        model.train(was_training)