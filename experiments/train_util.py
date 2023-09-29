from datetime import datetime
import pytz
from functools import partial
import torch
import torch.nn as nn
from timeit import default_timer
import time
import wandb

# [Qchem]
from nns.schnet.schnet import SchNetWrap
from nns.cgcnn.cgcnn import CGCNN
from nns.losses import EnergyForceLoss

# [Sciml]
from nns.unet.UnetDataSet import UNetDatasetSingle
from nns.unet.unetWrap import UNet1dWrap, unet_loss_fn

# [Vision]
from nns.vit.vit import Modifiedvit_b_16, Modifiedvit_b_16_adjust
from nns.cnn.cnn import CNN
from nns.resnet.resnet import ModifiedResNet18

# [Datasets]
from torch.utils.data import DataLoader
from data.MD17.MD17Dataset import MD17SingleDataset
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms


# =============================================================================
# Dataloader Collate
# =============================================================================

def unet_collate(batch, args):
    data, label = zip(*batch)
    
    data = torch.stack(data)
    label = torch.stack(label)
    
    x = data[..., 0, :]
    y = label[..., args.t_train - 1:args.t_train, :]
    
    x = x.permute([0, 2, 1])
    
    return x, y
    

def qchem_collate(batch):
    data = {"R":[], "z":[], "batch":[], "n":[]}
    label = {"E":[], "F":[], "covariance_inv":batch[0][4], "mean":batch[0][5]}      

    for i, b in enumerate(batch):
        data["R"].append(b[0])
        data["z"].append(b[1])
        data["n"].append(len(b[1]))
        data["batch"].append(torch.ones((b[1].size()), dtype=torch.int64)*i)
        label["E"].append(b[2])
        label["F"].append(b[3])
    data["R"] = torch.cat(data["R"])
    data["z"] = torch.cat(data["z"])
    data["n"] = torch.Tensor(data["n"])
    data["batch"] = torch.cat(data["batch"])
    label["E"] = torch.stack(label["E"])
    label["F"] = torch.stack(label["F"])
    
    data = {i: v for i, v in data.items()}
    label = {i: v for i, v in label.items()}
    
    return data, label


def vision_collate(batch):
    # Unzip the batch into data and labels lists
    data_list, label_list = zip(*batch)
    
    data = torch.stack(data_list)
    label = torch.tensor(label_list, dtype=torch.long)
        
    return data, label


# =============================================================================
# Model selection
# =============================================================================

def get_model_and_args(args):
    if args.nn_module == UNet1dWrap:
        return UNet1dWrap, [args.in_channels, args.out_channels]
    elif args.nn_module == SchNetWrap:
        return SchNetWrap, []
    elif args.nn_module == CGCNN:
        return CGCNN, []
    elif args.nn_module == CNN:
        return CNN, []
    elif args.nn_module == ModifiedResNet18:
        return ModifiedResNet18, []
    elif args.nn_module == Modifiedvit_b_16:
        return Modifiedvit_b_16, []
    elif args.nn_module == Modifiedvit_b_16_adjust:
        import numpy as np
        args.num_params = sum([np.prod(p.size()) for p in Modifiedvit_b_16_adjust(args.num_heads, args.num_layers, args.mlp_dim, args.hidden_dim).parameters()])
        print("PARAMETERS", args.num_params)
        return Modifiedvit_b_16_adjust, [args.num_heads, args.num_layers, args.mlp_dim, args.hidden_dim]
    else:
        raise ValueError(f"Model {args.nn_module} not supported ...")


def get_model(args):
    model = None
    if args.nn_module == UNet1dWrap:
        model  = UNet1dWrap(args.in_channels, args.out_channels)
    
    elif args.nn_module == SchNetWrap:
        model = SchNetWrap()
    
    elif args.nn_module == CGCNN:
        model = CGCNN()
    
    elif args.nn_module == CNN:
        model = CNN()

    elif args.nn_module == ModifiedResNet18:
        model = ModifiedResNet18()

    elif args.nn_module == Modifiedvit_b_16:
        model = Modifiedvit_b_16()

    elif args.nn_module == Modifiedvit_b_16_adjust:
        model = Modifiedvit_b_16_adjust(args.num_heads, args.num_layers, args.mlp_dim, args.hidden_dim)

    return model


# =============================================================================
# Loss function selection
# =============================================================================

def get_loss_fn(args):
    if args.model == "unet":
        return unet_loss_fn
    elif args.model == "schnet" or args.model == "cgcnn":
        return EnergyForceLoss
    elif args.model == "cnn" or  args.model == "resnet" or args.model == "transformer" or args.model == "transformer2":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Model {args.model} not supported ...")


# =============================================================================
# Dataloader selection
# =============================================================================


def get_dataloaders(args):
    print("\nGenerating Data Loaders.....")
    if args.model == "unet":
        file_name = '1D_Advection_Sols_beta0.1.hdf5'
        if args.cloud_path:
            base_path = '/home/paperspace/PusH2/experiments/data/1D/Advection/Train/'
        else:
            base_path = '/usr/data1/PDEBench/pdebench/data/1D/Advection/'
        train_dataset = UNetDatasetSingle(file_name,
                                          saved_folder=base_path,
                                          reduced_resolution=args.reduced_resolution,
                                          reduced_resolution_t=args.reduced_resolution_t,
                                          reduced_batch=args.reduced_batch,
                                          initial_step=args.initial_step)

        test_dataset = UNetDatasetSingle(file_name,
                                         saved_folder=base_path,
                                         reduced_resolution=args.reduced_resolution,
                                         reduced_resolution_t=args.reduced_resolution_t,
                                         reduced_batch=args.reduced_batch,
                                         initial_step=args.initial_step,
                                         if_test=True)
        
        if args.dataset_length is not None:
            train_dataset = torch.utils.data.random_split(train_dataset, [args.dataset_length, len(train_dataset) - args.dataset_length])[0]
            test_dataset = torch.utils.data.random_split(test_dataset, [int(args.dataset_length/10), len(test_dataset) - int(args.dataset_length/10)])[0]
            print("truncated train_dataset", len(train_dataset))

        # Create a decoy dataloader to update t_train 
        temp_train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=False)
        _, _data = next(iter(temp_train_loader))
        dimensions = len(_data.shape)
        print("Spatial Dimension", dimensions - 3)
        if args.t_train > _data.shape[-2]:
            args.t_train = _data.shape[-2]


        # collate_fn = partial(unet_collate, args=args, device=device)
        collate_fn = partial(unet_collate, args=args)
        # Build out dataloaders with pre processing 
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  num_workers=0, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 num_workers=0, shuffle=False, collate_fn=collate_fn)
        
        return train_loader, test_loader
    
    elif args.model == "schnet" or args.model == "cgcnn":
        train_dataset = MD17SingleDataset("trajectory", args.molecule, "train", args.split, root="./data/MD17")
        test_dataset = MD17SingleDataset("trajectory", args.molecule, "test", args.split, root="./data/MD17")
        
        if args.dataset_length is not None:
            train_dataset = torch.utils.data.random_split(train_dataset, [args.dataset_length, len(train_dataset) - args.dataset_length])[0]
            test_dataset = torch.utils.data.random_split(test_dataset, [int(args.dataset_length/10), len(test_dataset) - int(args.dataset_length/10)])[0]
            print("truncated train_dataset", len(train_dataset))
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=qchem_collate, num_workers=0)

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=qchem_collate, num_workers=0)
    
    elif args.model == "cnn" or args.model == "resnet":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB format
            # transforms.Resize((224, 224)),  # Resize to match ViT's expected input size
            transforms.ToTensor(),
        ])
        train_dataset = datasets.MNIST(root = "./data/vision/",train = True,transform = ToTensor(),
            download = True)
        test_dataset = datasets.MNIST(root = "./data/vision/", train = False, transform = ToTensor(),
            download=True)
        
        if args.dataset_length is not None:
            train_dataset = torch.utils.data.random_split(train_dataset, [args.dataset_length, len(train_dataset) - args.dataset_length])[0]
            test_dataset = torch.utils.data.random_split(test_dataset, [int(args.dataset_length/10), len(test_dataset) - int(args.dataset_length/10)])[0]
            print("truncated train_dataset", len(train_dataset))
        
        train_loader= DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                 num_workers=0,collate_fn=vision_collate)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                 collate_fn=vision_collate)
    
    elif args.model == "transformer" or args.model == "transformer2":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB format
            # transforms.Resize((224, 224)),  # Resize to match ViT's expected input size
            transforms.ToTensor(),
        ])
        train_dataset = datasets.MNIST(root = "./data/vision/",train = True,transform = transform,
            download = False)
        test_dataset = datasets.MNIST(root = "./data/vision/", train = False, transform = transform,
            download=False)
        
        if args.dataset_length is not None:
            train_dataset = torch.utils.data.random_split(train_dataset, [args.dataset_length, len(train_dataset) - args.dataset_length])[0]
            # test_dataset = torch.utils.data.random_split(test_dataset, [int(args.dataset_length/10), len(test_dataset) - int(args.dataset_length/10)])[0]
            print("truncated train_dataset", len(train_dataset))

        train_loader= DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader


# =============================================================================
# Dataloader selection
# =============================================================================

def get_argparser(parser):
    args, _ = parser.parse_known_args()    
    if args.model == "unet":
        # [Unet params]
        parser.add_argument("-nm", "--nn_module", type=nn.Module, default=UNet1dWrap)
        parser.add_argument("-rb", "--reduced_batch", type=int, default=4)
        parser.add_argument("-rr", "--reduced_resolution", type=int, default=4)
        parser.add_argument("-rrt", "--reduced_resolution_t", type=int, default=5)
        parser.add_argument("-is", "--initial_step", type=int, default=10)
        parser.add_argument("-tt", "--t_train", type=str, default=200)
        parser.add_argument("-nc", "--num_channels", type=int, default=1)
        parser.add_argument("-wi", "--in_channels", type=int, default=1)
        parser.add_argument("-mo", "--out_channels", type=int, default=1)
        parser.add_argument("-bs", "--batch_size", type=int, default=50)
        parser.add_argument("-us", "--unroll_step", type=int, default=20)
        parser.add_argument("-cp", "--cloud_path", action='store_true')
    
    elif args.model == "schnet":
        parser.add_argument("-nm", "--nn_module", type=nn.Module, default=SchNetWrap)
        parser.add_argument("-P", "--split", type=int, default=1000,
                            help="the name of dataset subset, aka the number of train samples")
        parser.add_argument("-m", "--molecule", default="a", type=str,
                            help="lowercase initial of the molecule in the dataset")
        parser.add_argument("-F", "--force", type=bool, default=True,
                            help="train with force")
        parser.add_argument("-b", "--batch_size", type=int, default=20,
                            help="batch size to train") 
    
    elif args.model == "cgcnn":
        parser.add_argument("-nm", "--nn_module", type=nn.Module, default=CGCNN)
        parser.add_argument("-P", "--split", type=int, default=1000,
                            help="the name of dataset subset, aka the number of train samples")
        parser.add_argument("-m", "--molecule", default="a", type=str,
                            help="lowercase initial of the molecule in the dataset")
        parser.add_argument("-F", "--force", type=bool, default=True,
                            help="train with force")
        parser.add_argument("-b", "--batch_size", type=int, default=20,
                            help="batch size to train") 
    
    elif args.model == "resnet":
        parser.add_argument("-nm", "--nn_module", type=nn.Module, default=ModifiedResNet18)
        parser.add_argument("-b", "--batch_size", type=int, default=128, help="batch size to train")
    
    elif args.model == "cnn":
        parser.add_argument("-nm", "--nn_module", type=nn.Module, default=CNN)
        parser.add_argument("-b", "--batch_size", type=int, default=128, help="batch size to train")
    
    elif args.model == "transformer":
        parser.add_argument("-nm", "--nn_module", type=nn.Module, default=Modifiedvit_b_16)
        parser.add_argument("-b", "--batch_size", type=int, default=128, help="batch size to train")
    
    elif args.model == "transformer2":
        parser.add_argument("-nm", "--nn_module", type=nn.Module, default=Modifiedvit_b_16_adjust)
        parser.add_argument("-b", "--batch_size", type=int, default=128, help="batch size to train")
        parser.add_argument("-num_heads", "--num_heads", type=int, default=8)
        parser.add_argument("-num_layers", "--num_layers", type=int, default=16)
        parser.add_argument("-mlp_dim", "--mlp_dim", type=int, default=1280)
        parser.add_argument("-hidden_dim", "--hidden_dim", type=int, default=320)
    
    if args.cache_size is None:
        args.cache_size = args.num_particles

    if args.view_size is None:
        args.view_size = args.cache_size

    return parser.parse_args()


def training_params(parser):
    args = get_argparser(parser)
    train_loader, test_loader = get_dataloaders(args)
    loss_fn = get_loss_fn(args)
    return args, train_loader, test_loader, loss_fn


# =============================================================================
# Wandb init
# =============================================================================

def wandb_init(args, dataloader):
    if args.wandb:
        timezone = pytz.timezone("America/Los_Angeles")
        start_dt = (datetime.now().astimezone(timezone)).strftime("%Y-%m-%d_%H-%M-%S")
        config = {
            "model": args.model,
            "train": args.train,
            "num_device": args.devices,
            "num_particles": args.num_particles,
            "cache_size": args.cache_size,
            "view_size": args.view_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "dataset size": len(dataloader.dataset),
        }
        if "num_params" in args:
            config["num_params"] = args.num_params
        print("Dataset size", len(dataloader.dataset))
        if args.train == args.train == "ensemble" or "ensemble_push":
            pass
        elif args.train == "mswag" or args.train == "mswag_push":
            config["pretrain_epochs"] = args.pretrain_epochs
            config["swag_epochs"] = args.swag_epochs
            config["num_models"] = args.num_models
            config["samples"] = args.samples
            config["scale"] = args.scale
            config["var_clamp"] = args.var_clamp
        elif args.train == "svgd" or args.train == "svgd_push":
            config["bandwidth"] = args.bandwidth

        wandb.init(
            project="scale_master", 
            entity="bogp", 
            group=args.group,
            name= f"{args.model}_{args.train}_{args.num_particles}_particles_{args.devices}_devices",
            id=start_dt, 
            config=config)
        

class MyTimer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = time.perf_counter() - self.start
        