import torch
import torch.nn as nn

class ETA_Module(nn.Module):
    """
    Early Temporal Aggregation (ETA) Module.
   
    """
    def __init__(self, pool_type='mean'):
        """
        Args:
            pool_type (str): Type of pooling. 'mean' for sequence mean pooling
                           (used for Transformer/BiLSTM outputs: [B, L, C]).
                           'gap' for global average pooling
                           (used for CNN outputs: [B, C, L]).
        """
        super().__init__()
        self.pool_type = pool_type

    def forward(self, x):
        """
        Args:
            x: Input tensor. Shape depends on pool_type:
               - If pool_type='mean': [Batch, Length, Channels]
               - If pool_type='gap':  [Batch, Channels, Length]
        Returns:
            Pooled tensor of shape [Batch, Channels]
        """
        if self.pool_type == 'mean':
            # Sequence Mean Pooling for Transformer/BiLSTM output
            # Input: [B, L, C] -> Output: [B, C]
            return x.mean(dim=1)
        elif self.pool_type == 'gap':
            # Global Average Pooling for CNN output
            # Input: [B, C, L] -> Output: [B, C]
            return x.mean(dim=2)
        else:
            raise ValueError(f"Unsupported pool_type: {self.pool_type}. Use 'mean' or 'gap'.")
