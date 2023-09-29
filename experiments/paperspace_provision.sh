#!/bin/bash

# Download anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh

# Install anaconda
bash Anaconda3-2022.05-Linux-x86_64.sh -b -p $HOME/anaconda3

# Initialize anaconda
./anaconda3/bin/conda init

# Create env
conda create -n push_exp python=3.10
conda activate push_exp
pip install pytz wandb matplotlib pandas torch torch_geometric torch_vision h5py pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install -e .

# Get datasets
cd experiments/data/MD17
python create_dataset.py
cd ../../..

cd experiments/data/vision
python create_dataset.py
cd ../../..

cd experiments/data/
python download_direct.py --root_folder /home/paperspace/push/experiments/data --pde_name advection	
