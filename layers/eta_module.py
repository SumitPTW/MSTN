import torch
import torch.nn as nn

class ETAModule(nn.Module):
    def __init__(self, d_model):
        super(ETAModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return x