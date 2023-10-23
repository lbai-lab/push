# PusH: Concurrent Probabilistic Programming for Bayesian Deep Learning

We introduce a library called PusH (**p**article p**ush**forward) that
enables a probabilistic programming approach to Bayesian deep learning (BDL).

1. Models are defined by wrapping ordinary `PyTorch` neural networks.
   Each particle corresponds to a point estimate. A collection of particles
   approximates a distributional estimate.
2. Inference procedures are defined as concurrent procedures on particles via message-passing.
3. Primary use case is BDL.

## Documentation
https://lbai-push.readthedocs.io/en/latest/


## Installation

1. Create an isolated Python environment
   `conda create -n push_exp python=3.10`
2. Install requirements
```
conda activate push_exp
pip install pytz wandb matplotlib pandas torch torch_geometric torch_vision h5py pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```
3. Locally install in project root.
   `pip install -e .`


## Quick Start

1. Install `push` and its dependencies following [installation](https://github.com/lbai-lab/push#installation).
2. Run some basic tests for various BDL algorithms. 
```
./run_tests.sh
```
3. Also experiment with:
  - `python ./test/test_basic.py -m ensemble`
  - `python ./test/test_basic.py -m mswag`
  - `python ./test/test_basic.py -m stein_vgd`


## Advanced Users

Add your own BDL algorithms in `/push/bayes` by extending the `Infer` class.
- Deep Ensembles, MultiSWAG, and Stein Variational Gradient Descent are implemented as examples.


## Experiments

1. See `./experiments/README.md` for more details.