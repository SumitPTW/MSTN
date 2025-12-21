import torch
import torch.nn as nn

class SGF_Module(nn.Module):
    """
    Self-Gated Fusion (SGF) Module.
    Learns to adaptively weight the concatenated features from dual pathways.
    Equation: z_fused = z_concat ⊙ σ(W_g · z_concat + b_g)
    """
    def __init__(self, fused_dim):
        super().__init__()
        # Linear layer for gating. Paper: W_g ∈ R^{192×192}
        self.gate = nn.Linear(fused_dim, fused_dim)

    def forward(self, z_concat):
        # z_concat shape: [Batch, Fused_Dim] (e.g., [B, 192])
        gate_weights = torch.sigmoid(self.gate(z_concat))
        z_fused = z_concat * gate_weights
        return z_fused
