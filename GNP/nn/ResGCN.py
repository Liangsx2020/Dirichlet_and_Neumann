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
# GCN with residual connections - Simplified and more robust.
class ResGCN(nn.Module):
    
    def __init__(self, A, num_layers, embed, hidden, drop_rate, device,
                 scale_input=True, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.num_layers = num_layers
        self.embed = embed
        self.scale_input = scale_input
        self.device = device
        self.n = A.shape[0]  # Store problem size

        # Move A to device and normalize
        self.AA = scale_A_by_spectral_radius(A).to(dtype).to(device)

        # Simplified input/output processing
        self.mlp_initial = MLP(1, embed, 3, hidden, drop_rate).to(device)  # Reduce layers
        self.mlp_final = MLP(embed, 1, 3, hidden, drop_rate,  # Reduce layers
                             is_output_layer=True).to(device)
        
        # GCN layers with simple residual connections
        self.gconv = nn.ModuleList()
        self.skip = nn.ModuleList()
        self.layer_norm = nn.ModuleList()  # Use LayerNorm for stability
        
        for i in range(num_layers):
            self.gconv.append(GCNConv(self.AA, embed, embed, device))
            self.skip.append(nn.Linear(embed, embed).to(device))
            self.layer_norm.append(nn.LayerNorm(embed).to(device))
            
        self.dropout = nn.Dropout(drop_rate)
        
        # Improved initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Small gain for stability
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, r):                        # r: (n, batch_size)
        assert len(r.shape) == 2
        n, batch_size = r.shape
        r = r.to(self.device)
        
        # Input scaling with better numerical stability
        if self.scale_input:
            scaling = torch.linalg.vector_norm(r, dim=0, keepdim=True) / np.sqrt(n)
            scaling = torch.clamp(scaling, min=1e-12)  # Better numerical stability
            r = r / scaling
        
        r = r.view(n, batch_size, 1)
        R = self.mlp_initial(r)                     # (n, batch_size, embed)
        
        # Simplified GCN layers with residual connections
        for i in range(self.num_layers):
            R_input = R
            
            # GCN transformation with residual
            R_gcn = self.gconv[i](R)
            R_skip = self.skip[i](R)
            R = R_gcn + R_skip  # Simple residual connection
            
            # Add deeper residual every few layers
            if i > 0 and i % 2 == 1:
                R = R + R_input
                
            # Normalization and activation
            R = R.view(n * batch_size, self.embed)
            R = self.layer_norm[i](R)
            R = R.view(n, batch_size, self.embed)
            R = self.dropout(F.relu(R))  # Use ReLU for simplicity
            
        z = self.mlp_final(R)                       # (n, batch_size, 1)
        z = z.view(n, batch_size)
        
        if self.scale_input:
            z = z * scaling.squeeze(0)  # Scale back
        return z