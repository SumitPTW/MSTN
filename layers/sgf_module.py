import torch
import torch.nn as nn

class SGF_Module(nn.Module):
    def __init__(self, fused_dim):  
        super().__init__()
        self.gate = nn.Linear(fused_dim, fused_dim) 
    
    def forward(self, z_concat):
        gate_weights = torch.sigmoid(self.gate(z_concat))
        return z_concat * gate_weights 
