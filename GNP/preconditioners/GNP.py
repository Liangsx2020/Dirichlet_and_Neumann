import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
import os
import numpy as np
from tqdm import tqdm

from GNP.solver import Arnoldi

    
#-----------------------------------------------------------------------------
# The following class implements a streaming dataset, which, in
# combined use with the dataloader, produces x of size (n,
# batch_size). x is float64 and stays in cpu. It will be moved to the
# device and cast to a lower precision for training.
class StreamingDataset(IterableDataset):

    # A is torch tensor, either sparse or full
    def __init__(self, A, batch_size, training_data, m):
        super().__init__()
        self.n = A.shape[0]
        self.m = m
        self.batch_size = batch_size
        self.training_data = training_data

        # Computations done in device
        if training_data == 'x_subspace' or training_data == 'x_mix':
            arnoldi = Arnoldi()
            Vm1, barHm = arnoldi.build(A, m=m)
            W, S, Zh = torch.linalg.svd(barHm, full_matrices=False)
            Q = ( Vm1[:,:-1] @ Zh.T ) / S.view(1, m)
            self.Q = Q.to('cpu')

    def generate(self):
        while True:

            # Computation done in cpu
            if self.training_data == 'x_normal':
                
                x = torch.normal(0, 1, size=(self.n, self.batch_size),
                                 dtype=torch.float64)
                yield x

            elif self.training_data == 'x_subspace':

                e = torch.normal(0, 1, size=(self.m, self.batch_size),
                                 dtype=torch.float64)
                x = self.Q @ e
                yield x

            elif self.training_data == 'x_mix':
            
                batch_size1 = self.batch_size // 3  # Reduce to 1/3 for subspace
                e = torch.normal(0, 1, size=(self.m, batch_size1),
                                 dtype=torch.float64)
                x1 = self.Q @ e
                
                batch_size2 = self.batch_size // 3  # 1/3 for normal random
                x2 = torch.normal(0, 1, size=(self.n, batch_size2),
                                  dtype=torch.float64)
                
                # Add multi-frequency training data for remaining batch
                batch_size3 = self.batch_size - batch_size1 - batch_size2
                x3 = self._generate_multi_frequency_data(batch_size3)
                
                x = torch.cat([x1, x2, x3], dim=1)
                yield x

            else: # self.training_data == 'no_x'

                b = torch.normal(0, 1, size=(self.n, self.batch_size),
                                 dtype=torch.float64)
                yield b
            
    def _generate_multi_frequency_data(self, batch_size):
        """Generate training data with multiple frequency components"""
        if batch_size <= 0:
            return torch.empty(self.n, 0, dtype=torch.float64)
            
        # Assume 2D grid problem for now
        n_side = int(np.sqrt(self.n))
        if n_side * n_side != self.n:
            # Fallback to random data if not a square grid
            return torch.normal(0, 1, size=(self.n, batch_size), dtype=torch.float64)
        
        x_batch = []
        for _ in range(batch_size):
            # Generate random frequencies
            freq_x = np.random.uniform(0.5, 8.0)  # Multiple frequency scales
            freq_y = np.random.uniform(0.5, 8.0)
            phase_x = np.random.uniform(0, 2*np.pi)
            phase_y = np.random.uniform(0, 2*np.pi)
            
            # Create 2D sinusoidal pattern
            x_coords = torch.linspace(0, 1, n_side, dtype=torch.float64)
            y_coords = torch.linspace(0, 1, n_side, dtype=torch.float64)
            X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')
            
            pattern = (torch.sin(freq_x * 2 * np.pi * X + phase_x) * 
                      torch.sin(freq_y * 2 * np.pi * Y + phase_y))
            pattern = pattern.flatten()
            
            # Add some noise
            pattern += 0.1 * torch.normal(0, 1, size=pattern.shape, dtype=torch.float64)
            x_batch.append(pattern.unsqueeze(1))
        
        if x_batch:
            return torch.cat(x_batch, dim=1)
        else:
            return torch.empty(self.n, 0, dtype=torch.float64)

    def __iter__(self):
        return iter(self.generate())


#-----------------------------------------------------------------------------
# Graph neural preconditioner
class GNP():

    # A is torch tensor, either sparse or full
    def __init__(self, A, training_data, m, net, device):
        self.A = A.to(device)
        self.training_data = training_data
        self.m = m
        self.net = net.to(device)  # Move net to device
        self.device = device
        self.dtype = net.dtype

    def train(self, batch_size, grad_accu_steps, epochs, optimizer,
              scheduler=None, num_workers=4, checkpoint_prefix_with_path=None,
              progress_bar=True):

        self.net.train()
        optimizer.zero_grad()
        dataset = StreamingDataset(self.A, batch_size,
                                   self.training_data, self.m)
        loader = DataLoader(dataset, num_workers=num_workers, pin_memory=True)
        
        hist_loss = []
        best_loss = np.inf
        best_epoch = -1
        checkpoint_file = None
            
        if progress_bar:
            pbar = tqdm(total=epochs, desc='Train')

        for epoch, x_or_b in enumerate(loader):

            # Generate training data
            if self.training_data != 'no_x':
                x = x_or_b[0].to(self.device)
                b = self.A @ x
                b, x = b.to(self.dtype), x.to(self.dtype)
            else: # self.training_data == 'no_x'
                b = x_or_b[0].to(self.device).to(self.dtype)

            # Train - Improved loss function for preconditioning
            x_out = self.net(b)
            # Focus on preconditioning equation: we want M(b) ≈ inv(A)b = x
            if self.training_data != 'no_x':
                # For training data where we know x, use direct loss
                loss = F.mse_loss(x_out, x)  # Use MSE for better gradient signal
            else:
                # For unknown x, use residual-based loss
                b_out = (self.A @ x_out.to(torch.float64)).to(self.dtype)
                loss = F.mse_loss(b_out, b)

            # Bookkeeping
            hist_loss.append(loss.item())
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch = epoch
                if checkpoint_prefix_with_path is not None:
                    checkpoint_file = checkpoint_prefix_with_path + 'best.pt'
                    torch.save(self.net.state_dict(), checkpoint_file)

            # Train (cont.)
            loss.backward()
            if (epoch+1) % grad_accu_steps == 0 or epoch == epochs - 1:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    # Handle different scheduler types
                    if hasattr(scheduler, 'step') and 'metrics' in scheduler.step.__code__.co_varnames:
                        scheduler.step(loss.item())  # For ReduceLROnPlateau
                    else:
                        scheduler.step()  # For other schedulers

            # Bookkeeping (cont.)
            if progress_bar:
                pbar.set_description(f'Train loss {loss:.1e}')
                pbar.update()
            if epoch == epochs - 1:
                break

        # Bookkeeping (cont.)
        if checkpoint_file is not None:
            checkpoint_file_old = checkpoint_file
            checkpoint_file = \
                checkpoint_prefix_with_path + f'epoch_{best_epoch}.pt'
            os.rename(checkpoint_file_old, checkpoint_file)
            
        return hist_loss, best_loss, best_epoch, checkpoint_file

    @torch.no_grad()
    def apply(self, r): # r: float64
        self.net.eval()
        r = r.to(self.device).to(self.dtype)  # Move to device and convert dtype
        r = r.view(-1, 1)
        z = self.net(r)
        z = z.view(-1)
        z = z.double() # -> float64
        return z