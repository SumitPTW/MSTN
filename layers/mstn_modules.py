import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleCNN(nn.Module):
    def __init__(self, c_in, d_model):
        super(MultiScaleCNN, self).__init__()
        self.conv1 = nn.Conv1d(c_in, 128, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=5, padding=2)

    def forward(self, x):
        # [B, L, C] -> [B, C, L]
        x = F.relu(self.bn1(self.conv1(x.transpose(1, 2))))
        x = F.relu(self.conv2(x))
        return x.transpose(1, 2) # [B, L, 64]

class SGFRecalibration(nn.Module):
    def __init__(self, d_model):
        super(SGFRecalibration, self).__init__()
        self.gate_weight = nn.Linear(d_model, d_model)
        self.se_fc1 = nn.Linear(d_model, 24)
        self.se_fc2 = nn.Linear(24, d_model)
        self.mha_recal = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.ln_final = nn.LayerNorm(d_model)

    def forward(self, z_concat):
        gate = torch.sigmoid(self.gate_weight(z_concat))
        z_fused = z_concat * gate
        z_se = z_fused * torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(z_fused))))
        z_mha, _ = self.mha_recal(z_se, z_se, z_se)
        return self.ln_final(z_mha)