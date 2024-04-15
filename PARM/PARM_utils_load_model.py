"""
Copyright 2024 PARM developers
https://github.com/vansteensellab/PARM

This file is part of PARM. PARM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version. PARM is distributed
in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along with PARM.
If not, see <http://www.gnu.org/licenses/>.
"""

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F


def load_PARM(weight_file: str):
    """
    Function to load the PARM model given a weight file
    """
    model = ResNet_Attentionpool()
    model_weights = torch.load(weight_file, map_location=torch.device("cpu"))
    model.load_state_dict(model_weights)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def ConvBlock(dim, dim_out=None, kernel_size=1):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding=kernel_size // 2),
    )


class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
        # (n p ) are length of sequence
        self.pool_fn = Rearrange("b d (n p) -> b d n p", p=pool_size)
        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0
        if needs_padding:
            x = F.pad(x, (0, remainder), value=0)
            mask = torch.zeros((b, 1, n), dtype=torch.bool, device=x.device)
            mask = F.pad(mask, (0, remainder), value=True)
        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)
        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)
        attn = logits.softmax(dim=-1)
        return (x * attn).sum(dim=-1)


class ResNet_Attentionpool(nn.Module):
    def __init__(self):
        super(ResNet_Attentionpool, self).__init__()

        self.L_max = 600
        self.vocab = 4
        self.type_loss = "poisson"
        kernel_size = 7
        stem_kernel_size = 7
        self.n_blocks = 5
        output_nodes = 1
        filter_size = 125

        ##################
        # create stem
        self.stem = nn.Sequential(
            nn.Conv1d(self.vocab, filter_size, stem_kernel_size, padding="same"),
            Residual(ConvBlock(filter_size)),
            AttentionPool(filter_size, pool_size=2),
        )

        # create conv tower
        conv_layers = []

        initial_filter_size = filter_size
        prev_filter_size = filter_size
        for block in range(self.n_blocks):
            if block > 4:
                filter_size = int(initial_filter_size * 0.2)

            conv_layers.append(
                nn.Sequential(
                    ConvBlock(prev_filter_size, filter_size, kernel_size=kernel_size),
                    Residual(ConvBlock(filter_size, filter_size, kernel_size=1)),
                    AttentionPool(filter_size, pool_size=2),
                )
            )

            prev_filter_size = filter_size

        self.conv_tower = nn.Sequential(*conv_layers)
        self.linear1 = nn.LazyLinear(out_features=output_nodes)
        self.relu = nn.ReLU()

        #################

    def forward(self, x):
        out = self.stem(x)
        out = self.conv_tower(out)
        out = torch.max(out, dim=-1).values
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.relu(out)

        return out
