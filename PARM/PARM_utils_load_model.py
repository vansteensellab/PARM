import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F


def load_PARM(
    weight_file: str = "_",
    L_max: int = 600,
    n_block: int = 5,
    filter_size: int = 125,
    train: bool = False,
    type_loss: str = 'poisson',
):
    """
    Function to load the PARM model given a weight file.
    
    Parameters
    ----------
    weight_file : str
        Path to the PARM model weights.
    L_max : int
        Maximum length of the input sequence.
    n_block : int
        Number of blocks in the model.
    filter_size : int
        Filter size for the model.
    train : bool
        Whether to return the model for training or inference.
        
    Returns
    -------
    model : nn.Module
        The PARM model.
    
    Examples
    --------
    >>> model = load_PARM("model.parm")
    """
    
    if train:
        model = ResNet_Attentionpool(L_max=L_max, n_block=n_block, filter_size=filter_size, weight_file=weight_file, type_loss=type_loss)
        if torch.cuda.is_available():
            model = model.cuda()
        return model
    else:
        model = ResNet_Attentionpool(L_max=L_max, n_block=n_block, filter_size=filter_size, weight_file=weight_file, type_loss=type_loss, validation=True)
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

    def __init__(self, L_max, n_block, filter_size=125, weight_file=None, type_loss='poisson', validation=False):
        super(ResNet_Attentionpool, self).__init__()

        self.type_loss = type_loss
        self.validation = validation

        self.L_max = L_max  # Max length of sequence
        self.vocab = 4  # N nucleotides

        kernel_size = 7
        stem_kernel_size = 7

        self.n_blocks = n_block

        # if '_TSS_EnhA_' in weight_file: filter_size = int(filter_size*2.4)

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

        if self.type_loss == 'heteroscedastic':
            self.log_var = nn.Linear(filter_size, 1)
            
        self.linear1 = nn.Linear(filter_size, 1)
        self.relu = nn.ReLU()

        #################

    def forward(self, x):

        out = self.stem(x)

        out = self.conv_tower(out)

        out = torch.max(out, dim=-1).values

        out = out.view(out.size(0), -1)

        out = self.linear1(out)

        if self.type_loss == 'heteroscedastic':
            mu = self.linear1(out)
            log_var = self.log_var(out)  # Log variance
            #return(mu)
            if self.validation: return mu
            return mu, log_var

        if self.type_loss == 'poisson': out = self.relu(out)

        return out
