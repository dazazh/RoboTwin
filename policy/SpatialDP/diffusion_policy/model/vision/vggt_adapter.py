import torch
import torch.nn as nn

class VGGTAdapter(nn.Module):
    def __init__(self, output_dim):
        super(VGGTAdapter, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(8 * 2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        # x: [batch_size, 8, 2048]
        x_flat = x.view(x.size(0), -1)  # -> [batch_size, 8*2048]
        x_out = self.mlp(x_flat)  # -> [batch_size, output_dim]
        return x_out
