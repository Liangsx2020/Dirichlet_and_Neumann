import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.xpu import device

from GNP.utils import scale_A_by_spectral_radius


#-----------------------------------------------------------------------------
# An MLP layer.
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, num_layers, hidden, drop_rate,
                 use_batchnorm=False, is_output_layer=False):
        super().__init__()
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.is_output_layer = is_output_layer

        self.lin = nn.ModuleList()
        self.lin.append( nn.Linear(in_dim, hidden) )
        for i in range(1, num_layers-1):
            self.lin.append( nn.Linear(hidden, hidden) )
        self.lin.append( nn.Linear(hidden, out_dim) )
        if use_batchnorm:
            self.batchnorm = nn.ModuleList()
            for i in range(0, num_layers-1):
                self.batchnorm.append( nn.BatchNorm1d(hidden) )
            if not is_output_layer:
                self.batchnorm.append( nn.BatchNorm1d(out_dim) )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, R):                              # R: (*, in_dim)
        assert len(R.shape) >= 2
        for i in range(self.num_layers):
            R = self.lin[i](R)                            # (*, hidden)
            if i != self.num_layers-1 or not self.is_output_layer:
                if self.use_batchnorm:
                    shape = R.shape
                    R = R.view(-1, shape[-1])
                    R = self.batchnorm[i](R)
                    R = R.view(shape)
                R = self.dropout(F.relu(R))
                                                          # (*, out_dim)
        return R
    

# -----------------------------------------------------------------------------
# A GCN layer.
class GCNConv(nn.Module):

    def __init__(self, AA, in_dim, out_dim, device):
        super().__init__()
        self.AA = AA.to(device)  # normalized A
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, R):                         # R: (n, batch_size, in_dim)
        assert len(R.shape) == 3
        n, batch_size, in_dim = R.shape
        assert in_dim == self.in_dim
        if in_dim > self.out_dim:
            R = self.fc(R)                           # (n, batch_size, out_dim)
            R = R.view(n, batch_size * self.out_dim) # (n, batch_size * out_dim)
            R = self.AA @ R                          # (n, batch_size * out_dim)
            R = R.view(n, batch_size, self.out_dim)  # (n, batch_size, out_dim)
        else:
            R = R.view(n, batch_size * in_dim)       # (n, batch_size * in_dim)
            R = self.AA @ R                          # (n, batch_size * in_dim)
            R = R.view(n, batch_size, in_dim)        # (n, batch_size, in_dim)
            R = self.fc(R)                           # (n, batch_size, out_dim)
        return R


# -----------------------------------------------------------------------------
# GCN with residual connections.
class ResGCN(nn.Module):
    
    def __init__(self, A, num_layers, embed, hidden, drop_rate, device,
                 scale_input=True, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.num_layers = num_layers
        self.embed = embed
        self.scale_input = scale_input
        self.device = device

        # Move A to device and normalize
        self.AA = scale_A_by_spectral_radius(A).to(dtype).to(device)

        # Move all modules to device
        self.mlp_initial = MLP(1, embed, 4, hidden, drop_rate).to(device)
        self.mlp_final = MLP(embed, 1, 4, hidden, drop_rate,
                             is_output_layer=True).to(device)
        self.gconv = nn.ModuleList()
        self.skip = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        for i in range(num_layers):
            self.gconv.append(GCNConv(self.AA, embed, embed, device))
            self.skip.append(nn.Linear(embed, embed).to(device))
            self.batchnorm.append(nn.BatchNorm1d(embed).to(device))
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, r):                        # r: (n, batch_size)
        assert len(r.shape) == 2
        n, batch_size = r.shape
        r = r.to(self.device)  # Move input to device
        
        if self.scale_input:
            scaling = torch.linalg.vector_norm(r, dim=0) / np.sqrt(n)
            r = r / scaling  # scaling
        r = r.view(n, batch_size, 1)                # (n, batch_size, 1)
        R = self.mlp_initial(r)                     # (n, batch_size, embed)
        
        for i in range(self.num_layers):
            R = self.gconv[i](R) + self.skip[i](R)  # (n, batch_size, embed)
            R = R.view(n * batch_size, self.embed)  # (n * batch_size, embed)
            R = self.batchnorm[i](R)                # (n * batch_size, embed)
            R = R.view(n, batch_size, self.embed)   # (n, batch_size, embed)
            R = self.dropout(F.relu(R))             # (n, batch_size, embed)
            
        z = self.mlp_final(R)                       # (n, batch_size, 1)
        z = z.view(n, batch_size)                   # (n, batch_size)
        if self.scale_input:
            z = z * scaling  # scaling back
        return z