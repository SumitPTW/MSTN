
import torch
import torch.nn as nn
import torch.nn.functional as F

class SE_Block(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced_channels = channels // reduction
        self.fc1 = nn.Linear(channels, reduced_channels)  # W₁
        self.fc2 = nn.Linear(reduced_channels, channels)  # W₂

    def forward(self, x):
        # x shape: [Batch, 1, Channels] (after ETA, temporal dim is 1)
        b, _, c = x.shape
        squeezed = x.mean(dim=1)  # Global average pooling across time
        excitation = torch.sigmoid(self.fc2(F.relu(self.fc1(squeezed))))
        return x * excitation.view(b, 1, c)
