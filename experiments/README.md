# PusH Experiments: 

Testing performance of PusH on various networks.


## Install

1. Create an environment
```
conda create -n push_exp python=3.10
conda activate push_exp
pip install pytz wandb matplotlib pandas torch torch_geometric torch_vision h5py pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
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
3. Run experiments

```
python train_qchem.py -model schnet -t swag_push
python train_qchem.py -model schnet -t svgd_push

python train_qchem.py -model cgcnn -t swag_push
python train_qchem.py -model cgcnn -t svgd_push
```


## Usage

1. See `paperspace_train_all_devices_1.sh`, `paperspace_train_all_devices_2.sh`, `paperspace_train_all_devices_3.sh`
   to test networks.
2. See `size_scale.sh`, `size_scale2.sh`, `size_scale4.sh` to test scaling.
3. See `train_baseline.sh` to test baseline implementations.


## Models

1. UNET: https://arxiv.org/pdf/1505.04597.pdf
2. Schnet: https://arxiv.org/pdf/1706.08566.pdf
3. CGCNN: https://arxiv.org/pdf/1710.10324.pdf
4. ResNet: https://arxiv.org/abs/1512.03385
5. Vision Transformer: https://arxiv.org/abs/2010.11929


