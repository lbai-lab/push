import os
import glob
import tqdm
import torch
import pickle
import requests
import argparse
import numpy as np

from torch.utils.data import Dataset


# Parent class of MD17 
class MD17Dataset(Dataset):
    def __init__(self, style, task, split, root="./"):
        # Names
        self.Mdict = {"a":"aspirin", "b":"benzene", "e":"ethanol", "m":"malonaldehyde", "n":"naphthalene", "s":"salicylic", "t":"toluene", "u":"uracil"}
        self.Ndict = {"a":21, "b":12, "e":9, "m":9, "n":18, "s":16, "t":15, "u":12}

        # Download missing data
        missing = [v for v in self.Mdict.values() if not os.path.isfile(f"{root}/{v}_dft.npz")]
        if missing: self.download(root, missing) 
        
        datas_dict, labels_dict = self.load_dataset(root, "*", style, task, split)
        self.datas, self.labels = self.unroll_data(datas_dict, labels_dict)
        self.covariance = self.cov_matrix(self.labels)

    def __len__(self):
        return len(self.datas["R"])

    def __getitem__(self, idx):
        return self.datas["R"][idx], self.datas["z"], self.labels["E"][idx], self.labels["F"][idx], self.labels["covariance_inv"], self.labels["mean"]

    def collate(self, batch):
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

        return data, label

    def print_info(self):
        print("Data sizes:")
        for m in self.Mdict.values():
            R = "R"
            print(f"{m}: {len(self.datas[m][R])}")
        return

    def download(self, path, molecules):
        print("Downloading missing datas")
        for m in tqdm.tqdm(molecules):
            if m == "benzene":
                r = requests.get(f"http://www.quantum-machine.org/gdml/data/npz/md17_{m}2017.npz")
            else:
                r = requests.get(f"http://www.quantum-machine.org/gdml/data/npz/md17_{m}.npz")
            open(os.path.join(path, f"{m}_dft.npz") , "wb").write(r.content)

        return 

    def load_dataset(self, root, style, molecule, task, split):
        npzs = glob.glob(f"{root}/{molecule}.npz")
        datas, labels = dict(), dict()

        for npz in npzs:
            z = np.load(npz, allow_pickle=True)
            name = npz.split("/")[-1].split(".")[0]
            datas[name], labels[name] = dict(), dict()

            with open(f"{root}/{split}/{molecule}/{task}.pkl", "rb") as f:
                idx = pickle.load(f)

            # Keys are ['E', 'name', 'F', 'theory', 'R', 'z', 'type', 'md5']
            datas[name]["R"] = z["R"][idx]
            datas[name]["z"] = z["z"]
            labels[name]["E"] = z["E"][idx]
            labels[name]["F"] = z["F"][idx]
        return datas, labels

    def cov_mean_matrix(self, label):
        E = label["E"]
        F = label["F"]
        F = F.reshape((F.size()[0],-1))
        EF = torch.t(torch.cat((E,F), axis=1))

        self.covariance_inv = torch.inverse(torch.cov(EF))
        self.mean = torch.mean(EF, dim=1)
        self.labels["covariance_inv"] = self.covariance_inv
        self.labels["mean"] = self.mean

        return 
    
    def unroll_data(self, datas_dict, labels_dict):
        datas, labels = dict(), dict()
        #TBD

        return datas, labels
    

class MD17SingleDataset(MD17Dataset):
    def __init__(self, style,  m, task, split, root="./"):
        # Initialize
        self.Mdict = {"a":"aspirin", "b":"benzene", "e":"ethanol", "m":"malonaldehyde", "n":"naphthalene", "s":"salicylic", "t":"toluene", "u":"uracil"} # full name of molecule
        self.Ndict = {"a":21, "b":12, "e":9, "m":9, "n":18, "s":16, "t":15, "u":12} # number of atom in each molecule
        self.identifier = f"MD17SingleDataset_{style}_{m}{split}{task}" # name tag for this dataset configuration
        self.m = self.Mdict[m]
        self.natom = self.Ndict[m]
                
        # Load and organized data
        datas_dict, labels_dict = self.load_dataset(root, style, self.m, task, split)
        self.datas, self.labels = self.unroll_data(datas_dict, labels_dict)
        self.cov_mean_matrix(self.labels)

    def print_info(self):
        print("Data sizes:")
        R = "R"
        print(f"{self.m}: {len(self.datas[R])}")
        return

    def unroll_data(self, datas_dict, labels_dict):
        datas = {"R":torch.Tensor(datas_dict[self.m]["R"]), "z":torch.LongTensor(datas_dict[self.m]["z"])} 
        labels = {"E":torch.Tensor(labels_dict[self.m]["E"]), "F":torch.Tensor(labels_dict[self.m]["F"])}

        return datas, labels


def parse_index(exps, l):
    indice = []
    
    for exp in exps.split(","):
        if exp == ":": indice += [i for i in range(l)]
        elif ":" not in exp:  indice += [int(exp)]
        else:
            if exp[0] == ":": indice += [i for i in range(int(exp[1:])+1)]
            elif exp[-1] == ":": indice += [i for i in range(int(exp[:-1])+1)]
            else: indice += [i for i in range(int(exp.split(":")[0]), int(exp.split(":")[1])+1)]

    return indice


def main(args):
    # As a script it could be used to display values in the dataset
    
    # Load dataset
    dataset = MD17SingleDataset(args.style, args.molecule, args.task, args.split)
    if args.info: dataset.print_info()
    
    # Get numerical index
    try:
        indice = parse_index(args.index, len(dataset.labels["E"]))
    except BaseException as err:
        print(err)
        print("expression of index incorrect")

    # Print
    R, z, E, F = "R", "z", "E", "F" 
    #print(f"covariance_inv:{dataset.covariance_inv}")
    #print(len(dataset.datas[z]))
    for idx in indice:
        print(f"{idx:06d}:\n (R={dataset.datas[R][idx]},\t z={dataset.datas[z]}),\t E={dataset.labels[E][idx]},\t F={dataset.labels[F][idx]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--root", type=str, help="path to the root dir of dataset", default="./")
    parser.add_argument("-y", "--style", type=str, default="trajectory", help="specify how to split the data")
    parser.add_argument("-m", "--molecule", type=str, help="lowercase initial of the molecule in the dataset", required=True)
    parser.add_argument("-t", "--task", type=str, help="train or valid or test", required=True)
    parser.add_argument("-s", "--split", type=int, help="the name of dataset subset, aka the number of train samples", required=True)
    parser.add_argument("-i", "--index", type=str, help="indice of the data to be shown", default=":")
    parser.add_argument("-f", "--info", type=bool, help="print the information about this dataset", default=False)

    main(parser.parse_args())
