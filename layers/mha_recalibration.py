import torch.nn as nn

class MHA_Recalibration(nn.Module):
    """
    Multi-Head Attention for feature recalibration after ETA.
    Operates on the single-token representation (L=1) after Early Temporal Aggregation.
    As per paper: Functions as a sophisticated global feature recalibration layer
    with O(1) complexity since temporal dimension is already collapsed.
    """
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        self.ln = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [Batch, 1, d_model] (single token after ETA)
        Returns:
            Recalibrated features of shape [Batch, 1, d_model]
        """
        # Self-attention on the single token (global feature refinement)
        attn_out, _ = self.mha(x, x, x)
        # Residual connection + LayerNorm
        return self.ln(x + attn_out)
