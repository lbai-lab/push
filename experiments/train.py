import argparse
from functools import partial
import numpy as np
from typing import *

import push.bayes.ensemble
import push.bayes.swag
import push.bayes.stein_vgd

from infer.standard import train_standard
from infer.ensemble import train_deep_ensemble
from infer.mswag import train_mswag
from infer.stein_vgd import train_svgd
from infer.push_ensemble import _ensemble_main_instrumented, mk_optim
from infer.push_mswag import _mswag_particle_instrumented, _mswag_sample_entry_instrumented, _mswag_sample_instrumented
from infer.push_stein_vgd import _svgd_leader_instrumented, _svgd_leader_instrumented_memeff

from train_util import training_params, get_model, get_model_and_args


if __name__ == "__main__":

    # Initialize a list to store memory usage for each GPU
    parser = argparse.ArgumentParser()
    
    # [Training params]
    parser.add_argument("-wb", "--wandb", action='store_true')
    parser.add_argument("-g", "--group", type=str, default='default')
    parser.add_argument("-e", "--epochs", type=int, default=5)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-dl", "--dataset_length", type=int, default=None)
    parser.add_argument("-w", "--weight_decay", type=float, default=0)
    parser.add_argument("-ss", "--scheduler_step", type=int, default=100)
    parser.add_argument("-sg", "--scheduler_gamma", type=int, default=0.5)
    parser.add_argument("-model", "--model", type=str, default="vit",
                        choices=[
                            "schnet",
                            "cgcnn",
                            "unet",
                            "resnet",
                            "cnn",
                            "transformer",
                            "transformer2",
                        ])
    parser.add_argument("-t", "--train", type=str, default="svgd_push",
                        choices=[
                            "standard",
                            "ensemble",
                            "ensemble_push",
                            "mswag",
                            "mswag_push",
                            "svgd",
                            "svgd_push"
                        ])
    
    # [Push args]
    parser.add_argument("-n", "--num_particles", type=int, default=2)
    parser.add_argument("-cs", "--cache_size", type=int, default=None)
    parser.add_argument("-vs", "--view_size", type=int, default=None)
    parser.add_argument("-d", "--devices", type=int, default=1)
    parser.add_argument("-save", "--save", action="store_true")
    parser.add_argument("-ppred", "--posterior-pred", action="store_true")

    # [SWAG params]
    parser.add_argument("-pe", "--pretrain_epochs", type=int, default=1)
    parser.add_argument("-se", "--swag_epochs", type=int, default=1)
    parser.add_argument("-samples", "--samples", type=int, default=5)
    parser.add_argument("-scale", "--scale", type=float, default=1.0)
    parser.add_argument("-var_clamp", "--var_clamp", type=float, default=1e-30)

    # [SVGD params]
    parser.add_argument("-band", "--bandwidth", type=float, default=1)
    parser.add_argument("-mef", "--mem_eff", action="store_true")
    
    # Add training parameters
    args, train_loader, test_loader, loss_fn = training_params(parser)

    # GPU memory
    if args.cache_size is None:
        args.cache_size = int(np.ceil(args.num_particles/args.devices))
    if args.view_size is None:
        args.view_size = args.num_particles
        

    # Select
    if args.train == "standard":
        trained_model = train_standard(train_loader, args, loss_fn)

    elif args.train == "ensemble":
        trained_models = train_deep_ensemble(train_loader, args, loss_fn)

    elif args.train == "ensemble_push":
        model, model_args = get_model_and_args(args)
        ensemble_state = {
            "args": args,
        }
        push.bayes.ensemble.train_deep_ensemble(
            train_loader, loss_fn, args.epochs, model, *model_args,
            cache_size=args.cache_size, view_size=args.view_size, 
            num_ensembles=args.num_particles, num_devices=args.devices,
            mk_optim=partial(mk_optim, args.learning_rate, args.weight_decay),
            ensemble_entry=_ensemble_main_instrumented, ensemble_state=ensemble_state
        )
    
    elif args.train == "mswag":
        nns = train_mswag(train_loader, args, loss_fn)

    elif args.train == "mswag_push":
        model, model_args = get_model_and_args(args)
        if args.cache_size is None:
            args.cache_size = np.ceil(args.num_particles/args.devices)
        # args.view_size = args.num_particles
        mswag_state = { 
            "args": args,
        }
        mswag = push.bayes.swag.train_mswag(
            train_loader, loss_fn, args.pretrain_epochs, args.swag_epochs,
            args.num_particles, args.cache_size, args.view_size, 
            model, *model_args,
            num_devices=args.devices, lr=args.learning_rate,
            mswag_entry=_mswag_particle_instrumented, mswag_state=mswag_state,
            f_save=args.save, mswag_sample_entry=_mswag_sample_entry_instrumented, mswag_sample=_mswag_sample_instrumented
        )
        if args.posterior_pred:
            mswag.posterior_pred(test_loader, loss_fn, num_samples=args.samples , scale=args.scale, var_clamp=args.var_clamp)
    
    elif args.train == "svgd": 
        networks = train_svgd(train_loader, args, loss_fn) 

    elif args.train == "svgd_push":
        prior = None
        svgd_state = { 
            "args": args,
        }
        model, model_args = get_model_and_args(args)
        if args.mem_eff:
            svgd_entry = _svgd_leader_instrumented_memeff
        else:
            svgd_entry = _svgd_leader_instrumented
        push.bayes.stein_vgd.train_svgd(
            train_loader, loss_fn, args.epochs, args.num_particles, model, *model_args,
            num_devices=args.devices, cache_size=args.cache_size, view_size=args.view_size,
            lengthscale=args.bandwidth, lr=args.learning_rate, prior=prior,
            svgd_entry=svgd_entry, svgd_state=svgd_state
        )
    else:
        raise ValueError(f"Method {args.train} not supported ...")
