# PusH Experiments: 

Testing performance of PusH on various networks.


## Install

1. Create an environment
```
conda create -n push_exp python=3.10
conda activate push_exp
pip install pytz wandb matplotlib pandas torch torch_geometric torch_vision torchvision h5py pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install -e .
```
2. Download data
    - Get MD17
    ```
    cd experiments/data/MD17
    python create_dataset.py
    ```
    - Get MNIST
    ```
    cd experiments/data/vision
    python create_dataset.py
    ```
    - Get Advection
    ```
    cd experiments/data
    python download_direct.py --root_folder /home/paperspace/PusH2/experiments/data --pde_name advection
    ```
3. Quick test
```
./scripts/quick_test.sh
```

## Usage

1. See `./scripts/paperspace_train_all_devices_1.sh`, `./scripts/paperspace_train_all_devices_2.sh`, `./scripts/paperspace_train_all_devices_3.sh`
   to test scaling of particles across architectures, devices, and tasks.
2. See `./scripts/size_scale_1.sh`, `./scripts/size_scale_2.sh`, `./scripts/size_scale_4.sh` to test scaling of PusH on various transformers of various depths.
3. See `./scripts/train_baseline.sh` to test baseline implementations.
4. See `./scripts/bayes_1.sh` and `./scripts/bayes_2.sh` to test multi-SWAG on transformers of various depths and widths.


## Models

1. UNET: https://arxiv.org/pdf/1505.04597.pdf
2. Schnet: https://arxiv.org/pdf/1706.08566.pdf
3. CGCNN: https://arxiv.org/pdf/1710.10324.pdf
4. ResNet: https://arxiv.org/abs/1512.03385
5. Vision Transformer: https://arxiv.org/abs/2010.11929
