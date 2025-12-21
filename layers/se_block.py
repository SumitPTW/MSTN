import torch
import torch.nn as nn
import torch.nn.functional as F

class SE_Block(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced_channels = channels // reduction
        self.fc1 = nn.Linear(channels, reduced_channels)  
        self.fc2 = nn.Linear(reduced_channels, channels)  

    def forward(self, x):
        b, l, c = x.shape
        
        if l == 1:
            squeezed = x.squeeze(1)
        else:
            squeezed = x.mean(dim=1)
            
        excitation = torch.sigmoid(self.fc2(F.relu(self.fc1(squeezed))))
        return x * excitation.view(b, 1, c)
