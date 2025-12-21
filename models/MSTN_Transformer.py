import torch
import torch.nn as nn
from layers.mstn_modules import MultiScaleCNN, TransformerPathway
from layers.eta_module import ETA_Module
from layers.sgf_module import SGF_Module
from layers.se_block import SE_Block
from layers.mha_recalibration import MHA_Recalibration

class MSTN_Transformer(nn.Module):
    """
    MSTN with Transformer as the sequence modeling pathway.
    Implements the architecture described in Section 2.1 of the manuscript.
    
    The model follows the Early Temporal Aggregation (ETA) principle:
    1. Dual-path encoding (CNN + Transformer)
    2. ETA: Temporal pooling (L -> 1) for O(1) inference complexity
    3. Multi-scale fusion and refinement (SGF, SE, MHA)
    4. Task-specific prediction head
    """
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        
        # --- 1. Dual Pathways (Paper Section 2.1) ---
        # CNN pathway for local patterns
        self.cnn_pathway = MultiScaleCNN(configs.enc_in, cnn_hidden=64)
        # Transformer pathway for global dependencies
        self.seq_pathway = TransformerPathway(
            c_in=configs.enc_in,
            d_model=128,
            nhead=8,
            num_layers=4
        )
        
        # --- 2. Early Temporal Aggregation (ETA) ---
        # Core innovation for O(1) complexity
        self.eta_cnn = ETA_Module(pool_type='gap')    # Global Average Pooling for CNN
        self.eta_seq = ETA_Module(pool_type='mean')   # Sequence Mean Pooling for Transformer
        
        # --- 3. Fusion & Refinement ---
        fused_dim = 64 + 128  # cnn_hidden + d_model = 192
        self.sgf = SGF_Module(fused_dim)  # Self-Gated Fusion
        self.se_block = SE_Block(fused_dim, reduction=8)  # Squeeze-and-Excitation
        self.mha_recal = MHA_Recalibration(fused_dim, num_heads=4)
        self.dropout = nn.Dropout(configs.dropout)
        
        # --- 4. Task-Specific Heads (Critical) ---
        if self.task_name == 'classification':
            self.head = nn.Linear(fused_dim, configs.num_class)
        elif self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.head = nn.Linear(fused_dim, configs.pred_len * configs.c_out)
            self.pred_len = configs.pred_len
            self.c_out = configs.c_out
        elif self.task_name == 'imputation':
            self.head = nn.Linear(fused_dim, configs.seq_len * configs.c_out)
            self.seq_len = configs.seq_len
            self.c_out = configs.c_out
        else:
            raise ValueError(f"Unsupported task: {self.task_name}")
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward pass for MSTN-Transformer.
        
        Args:
            x_enc: Input tensor of shape [Batch, Sequence_Length, Channels]
            x_mark_enc: Optional temporal features (not used in base MSTN)
            x_dec: Decoder input (for forecasting tasks)
            x_mark_dec: Optional decoder temporal features
            mask: Optional mask for imputation tasks
        
        Returns:
            Task-specific output:
            - Classification: [Batch, num_class]
            - Forecasting: [Batch, pred_len, c_out]
            - Imputation: [Batch, seq_len, c_out]
        """
        # --- 1. Dual Pathway Processing ---
        # CNN path: expects [B, C, L]
        h_cnn = self.cnn_pathway(x_enc.transpose(1, 2))  # [B, 64, L]
        # Transformer path: expects [B, L, C]
        h_seq = self.seq_pathway(x_enc)  # [B, L, 128]
        
        # --- 2. Early Temporal Aggregation (L -> 1) ---
        z_cnn = self.eta_cnn(h_cnn)  # [B, 64]
        z_seq = self.eta_seq(h_seq)  # [B, 128]
        
        # --- 3. Fusion & Refinement ---
        z_concat = torch.cat([z_cnn, z_seq], dim=1)  # [B, 192]
        z_fused = self.sgf(z_concat)  # Self-Gated Fusion
        z_se = self.se_block(z_fused.unsqueeze(1))  # SE Block expects [B, 1, 192]
        z_mha = self.mha_recal(z_se)  # Multi-Head Attention recalibration
        z_final = self.dropout(z_mha.squeeze(1))  # [B, 192]
        
        # --- 4. Task-Specific Output ---
        out = self.head(z_final)
        
        # Reshape for sequence output tasks
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # Reshape to [B, H, C]
            return out.view(out.shape[0], self.pred_len, self.c_out)
        elif self.task_name == 'imputation':
            # Reshape to [B, L, C]
            return out.view(out.shape[0], self.seq_len, self.c_out)
        else:  # classification
            return out  # [B, num_class]
