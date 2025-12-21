import torch
import torch.nn as nn

class SGFModule(nn.Module):
    def __init__(self, channels):
        super(SGFModule, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, L, C]
        gate_weights = self.gate(x)
        return x * gate_weights