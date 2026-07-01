import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F


def load_PARM(
    weight_file: str = None,
    L_max: int = 600,
    n_block: int = 5,
    filter_size: int = 125,
    train: bool = False,
    type_loss: str = 'poisson',
    cell_line: str = False,
    maxglobalpool: bool = True,
    use_AttentionPool: bool = True
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
    type_loss : str
        Type of loss function used in the model ('poisson' or 'heteroscedastic').
    cell_line : str
        Cell line information for the model (if applicable).
    maxglobalpool : bool
        Whether to use max global pooling in the model.
    use_AttentionPool : bool
        Whether to use AttentionPool instead of MaxPool in the model.
        
    Returns
    -------
    model : nn.Module
        The PARM model.
    
    Examples
    --------
    >>> model = load_PARM("model.parm")
    """

    model = ResNet_Attentionpool(L_max=L_max, n_block=n_block, 
                                filter_size=filter_size, weight_file=weight_file, 
                                type_loss=type_loss, cell_line=cell_line,
                                validation=(not train),
                                index_interested_output=False, 
                                maxglobalpool=maxglobalpool,
                                use_AttentionPool=use_AttentionPool)
    
    if train:
        if torch.cuda.is_available():
            model = model.cuda()
        return model
    else:
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


def ConvBlock(dim, dim_out = None, kernel_size = 1):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding = kernel_size // 2)
    )


class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        #(n p ) are length of sequence
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)
        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias = False)
    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0
        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)
        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)
        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)
        attn = logits.softmax(dim = -1)
        return (x * attn).sum(dim = -1)

class DenseLayersAfterSplit(nn.Module):
    
    def __init__(self, filter_size, output_nodes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(filter_size, output_nodes),
            nn.ReLU(),
            nn.Linear(output_nodes, 1)
        )

    def forward(self, x):
        return self.net(x)
class ResNet_Attentionpool(nn.Module):

    def __init__(self, L_max, n_block, filter_size=125, weight_file=None, 
                cell_line=False,
                type_loss='poisson', validation=False, index_interested_output=False, maxglobalpool=True,
                vocab=4, use_AttentionPool=True,
                dense_layer_after_split=False,
                dense_layer_size=64):
        super(ResNet_Attentionpool, self).__init__()

        self.type_loss = type_loss
        if type_loss == 'heteroscedastic': self.heteroscedastic = True
        else: self.heteroscedastic = False
        self.index_interested_output = index_interested_output
        self.validation = validation
        self.maxglobalpool = maxglobalpool
        self.L_max = L_max  # Max length of sequence
        self.vocab = vocab  # N nucleotides
        self.dense_layer_after_split = dense_layer_after_split
        self.dense_layer_size = dense_layer_size

        kernel_size = 7
        stem_kernel_size = 7
        
        if cell_line:
            output_nodes = len(cell_line.split('__'))
        
        else:
            output_nodes = 1

        self.n_blocks = n_block

        ##################
        # create stem
        self.stem = nn.Sequential(
                    nn.Conv1d(vocab, filter_size, stem_kernel_size, padding = "same"),
                    Residual(ConvBlock(filter_size)),
                    AttentionPool(filter_size, pool_size = 2) if use_AttentionPool else nn.MaxPool1d(2) )


        # create conv tower
        conv_layers = []

        initial_filter_size = filter_size
        prev_filter_size = filter_size
        for block in range(self.n_blocks):
            if block > 4: 
                filter_size = int(initial_filter_size*0.2)
            
            conv_layers.append(nn.Sequential(
                ConvBlock(prev_filter_size, filter_size, kernel_size = kernel_size),
                Residual(ConvBlock(filter_size, filter_size, kernel_size = 1)),
                AttentionPool(filter_size, pool_size = 2) if use_AttentionPool else nn.MaxPool1d(2)
            ))
            
            prev_filter_size = filter_size

        self.conv_tower = nn.Sequential(*conv_layers)
        if self.dense_layer_after_split is False:
            self.linear1 = nn.Linear(filter_size, output_nodes)  # Mean output
            if self.heteroscedastic:
                self.log_var = nn.Linear(filter_size, output_nodes)  # Log-variance output
            self.relu = nn.ReLU()
        else:
            self.linear1 = nn.Linear(filter_size, output_nodes)  # shared layer
            self.cell_heads = nn.ModuleList([DenseLayersAfterSplit(filter_size, self.dense_layer_size) for _ in range(output_nodes)])  # a dense layer per cell line
            if self.heteroscedastic:
                self.log_var_heads = nn.ModuleList([DenseLayersAfterSplit(filter_size, self.dense_layer_size) for _ in range(output_nodes)])

        #################

    def forward(self, x):
        if self.dense_layer_after_split is False:
            out = self.stem(x)
            out = self.conv_tower(out)
        else:
            out = self.stem(x)
            out = self.conv_tower(out)
            out = torch.cat([head(out) for head in self.cell_heads], dim=-1)

        if self.maxglobalpool:
            #max in length
            out = torch.max(out, dim=-1).values

        out = out.view(out.size(0), -1)

        if self.heteroscedastic:
            mu = self.linear1(out)
            log_var = self.log_var(out)  # Log variance
            #return(mu)
            if self.validation: return mu
            return mu, log_var
        
        else:
            out = self.linear1(out)
        

        if self.type_loss == 'poisson': out = self.relu(out)
        if self.index_interested_output: out = out[:,self.index_interested_output].unsqueeze(1)


        return out
