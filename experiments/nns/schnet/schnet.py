"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import sys
import torch

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if proj_root not in sys.path: sys.path.insert(1, proj_root)
from torch_geometric.nn import SchNet
from nns.schnet.base import BaseModel

class SchNetWrap(SchNet, BaseModel):
    r"""Wrapper around the continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_. Each layer uses interaction
    block of the form:
    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),
    Args:
        num_atoms (int): Unused argument
        bond_feat_dim (int): Unused argument
        num_targets (int): Number of targets to predict.
        use_pbc (bool, optional): If set to :obj:`True`, account for periodic boundary conditions.
            (default: :obj:`True`)
        regress_forces (bool, optional): If set to :obj:`True`, predict forces by differentiating
            energy with respect to positions.
            (default: :obj:`True`)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly.
            (default: :obj:`False`)
        hidden_channels (int, optional): Number of hidden channels.
            (default: :obj:`128`)
        num_filters (int, optional): Number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): Number of interaction blocks
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
    """

    def __init__(
        self,
        regress_forces=True,
        hidden_channels=1024,
        num_filters=256,
        num_interactions=5,
        num_gaussians=200,
        cutoff=6.0,
        readout="add",):

        self.regress_forces = regress_forces
        self.cutoff = cutoff
        self.max_neighbors = 50
        self.freeze_layers = ['embedding.weight', 'interactions.0.mlp.0.weight', 'interactions.0.mlp.2.weight', 'interactions.0.conv.lin1.weight', 'interactions.0.conv.lin2.weight', 
                              'interactions.0.conv.lin2.bias', 'interactions.0.lin.weight', 'interactions.0.lin.bias', 'interactions.1.mlp.0.weight', 'interactions.1.mlp.2.weight', 
                              'interactions.1.conv.lin1.weight', 'interactions.1.conv.lin2.weight', 'interactions.1.conv.lin2.bias', 'interactions.1.lin.weight', 
                              'interactions.1.lin.bias', 'interactions.2.mlp.0.weight', 'interactions.2.mlp.2.weight', 'interactions.2.conv.lin1.weight', 
                              'interactions.2.conv.lin2.weight', 'interactions.2.conv.lin2.bias', 'interactions.2.lin.weight', 'interactions.2.lin.bias', 
                              'interactions.3.mlp.0.weight', 'interactions.3.mlp.2.weight', 'interactions.3.conv.lin1.weight', 'interactions.3.conv.lin2.weight', 
                              'interactions.3.conv.lin2.bias', 'interactions.3.lin.weight', 'interactions.3.lin.bias', 'interactions.4.conv.lin1.weight', 
                              'interactions.4.conv.lin2.weight', 'interactions.4.conv.lin2.bias', 'interactions.4.lin.weight', 'interactions.4.lin.bias', 
                              'lin1.weight', 'lin1.bias', 'lin2.bias']

        super(SchNetWrap, self).__init__(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout,)

    def forward(self, data):
        z = data["z"]
        pos = data["R"]
        batch = data["batch"]

        if self.regress_forces:
            pos.requires_grad_(True)
        energy = super(SchNetWrap, self).forward(z, pos, batch=batch)

        if self.regress_forces:
            forces = -1*(torch.autograd.grad(energy, pos, grad_outputs=torch.ones_like(energy), create_graph=True,)[0])
            return {"E":energy, "F":forces}
        else:
            return {"E":energy}
