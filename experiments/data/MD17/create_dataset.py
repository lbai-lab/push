from genericpath import isdir
import os
import random
import requests
import pickle
import shutil
import argparse
# from tqdm import tqdm


MOLECULES = {
    "a": "aspirin",
    "b": "benzene",
    "e": "ethanol",
    "m": "malonaldehyde",
    "n": "naphthalene",
    "s": "salicylic",
    "t": "toluene",
    "u": "uracil"
}


def download(path: str) -> None:
    print("Downloading missing data ...")
    missing = [v for v in MOLECULES.values() if not os.path.isfile(f"{path}/{v}.npz")]
    for m in (missing):
        print("m: ", m)
        if m == "benzene":
            r = requests.get(f"http://www.quantum-machine.org/gdml/data/npz/md17_{m}2017.npz")
        else:
            r = requests.get(f"http://www.quantum-machine.org/gdml/data/npz/md17_{m}.npz")
            # print(f"http://www.quantum-machine.org/gdml/data/npz/{m}.npz")
            print(r)
        open(os.path.join(path, f"{m}.npz") , "wb").write(r.content)


def split(args, path):
    sample_num = {
        "aspirin": 211762,
        "benzene": 627983,
        "ethanol": 555092,
        "malonaldehyde": 993237,
        "naphthalene": 326250,
        "salicylic": 320231,
        "toluene": 442790,
        "uracil": 133770
    }

    if os.path.isdir(f"{args.train}"): 
        print("over writing split")
        shutil.rmtree(f"./{args.train}")
    os.mkdir(f"{args.train}")

    for m in sample_num.keys():
        os.mkdir(f"{args.train}/{m}")
        assert args.train+args.valid < sample_num[m]

        pool = [i for i in range(sample_num[m])]
        train_set = random.sample(pool, args.train)
        pool = list(set(pool)-set(train_set))
        valid_set = random.sample(pool, args.valid)
        test_set = list(set(pool)-set(valid_set))
        assert len(train_set)+len(valid_set)+len(test_set) == sample_num[m]

        print(f"{args.train}/{m}/train.pkl")
        with open(f"{args.train}/{m}/train.pkl", "wb") as f:
            pickle.dump(train_set, f)
        with open(f"{args.train}/{m}/valid.pkl", "wb") as f:
            pickle.dump(valid_set, f)
        with open(f"{args.train}/{m}/test.pkl", "wb") as f:
            pickle.dump(test_set, f)

    print("split generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1000, help="number of sample in training set")
    parser.add_argument("--valid", type=int, default=100, help="number of sample in validation set")

    args = parser.parse_args()
    download(os.getcwd())
  
