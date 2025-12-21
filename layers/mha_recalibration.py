import torch.nn as nn

class MHA_Recalibration(nn.Module):
  
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        self.ln = nn.LayerNorm(d_model)
    
    def forward(self, x):
      
        # Self-attention on the single token (global feature refinement)
        attn_out, _ = self.mha(x, x, x)
        # Residual connection + LayerNorm
        return self.ln(x + attn_out)
