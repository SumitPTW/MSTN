import torch
import torch.nn as nn
import torch.nn.functional as F

class SGFModule(nn.Module):
    """
    Selective Gated Fusion (SGF) Module
    Recalibrates features by learning channel-wise dependencies through a bottleneck.
    """
    def __init__(self, channels, reduction=8):
        super(SGFModule, self).__init__()
        # Bottleneck architecture for efficiency
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):

        gate = F.relu(self.fc1(x))
        selection_mask = torch.sigmoid(self.fc2(gate))
        return x * selection_mask
