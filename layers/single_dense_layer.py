import torch.nn as nn

class SingleDenseLayer(nn.Module):
    """
    Single Dense Layer with Dropout and LayerNorm.
        
    This layer performs:
    1. Linear transformation (dim → dim)
    2. Dropout for regularization
    3. Layer Normalization
    
    Input: [B, 1, dim] or [B, dim]
    Output: [B, 1, dim]
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x shape: [B, 1, dim] or [B, dim]
        x = x.squeeze(1)           # [B, dim]
        x = self.linear(x)          # [B, dim]
        x = self.dropout(x)         # [B, dim]
        x = self.norm(x)            # [B, dim]
        return x.unsqueeze(1)       # [B, 1, dim]
