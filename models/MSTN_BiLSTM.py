import torch
import torch.nn as nn
from layers.mstn_modules import MultiScaleCNN, BiLSTMPathway  # Changed import
from layers.eta_module import ETA_Module
from layers.sgf_module import SGF_Module
from layers.se_block import SE_Block
from layers.mha_recalibration import MHA_Recalibration

class MSTN_BiLSTM(nn.Module):
    """
    MSTN with BiLSTM as the sequence modeling pathway.
    Implements the architecture described in Section 2.1 of the manuscript.
    
    The model follows the Early Temporal Aggregation (ETA) principle:
    1. Dual-path encoding (CNN + BiLSTM)
    2. ETA: Temporal pooling (L -> 1) for O(1) inference complexity
    3. Multi-scale fusion and refinement (SGF, SE, MHA)
    4. Task-specific prediction head
    """
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        
        # --- 1. Dual Pathways (Paper Section 2.1) ---
        # CNN pathway for local patterns (same as Transformer variant)
        self.cnn_pathway = MultiScaleCNN(configs.enc_in, cnn_hidden=64)
        # BiLSTM pathway for sequential dependencies
        self.seq_pathway = BiLSTMPathway(
            c_in=configs.enc_in,
            lstm_hidden=128,  # 64 per direction * 2 = 128 total
            num_layers=2
        )
        
        # --- 2. Early Temporal Aggregation (ETA) ---
        self.eta_cnn = ETA_Module(pool_type='gap')    # Global Average Pooling for CNN
        self.eta_seq = ETA_Module(pool_type='mean')   # Sequence Mean Pooling for BiLSTM
        
        # --- 3. Fusion & Refinement ---
        fused_dim = 64 + 128  # cnn_hidden + lstm_hidden = 192
        self.sgf = SGF_Module(fused_dim)
        self.se_block = SE_Block(fused_dim, reduction=8)
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
        # --- 1. Dual Pathway Processing ---
        # CNN path: expects [B, C, L]
        h_cnn = self.cnn_pathway(x_enc.transpose(1, 2))  # [B, 64, L]
        # BiLSTM path: expects [B, L, C]
        h_seq = self.seq_pathway(x_enc)  # [B, L, 128]
        
        # --- 2. Early Temporal Aggregation (L -> 1) ---
        z_cnn = self.eta_cnn(h_cnn)  # [B, 64]
        z_seq = self.eta_seq(h_seq)  # [B, 128]
        
        # --- 3. Fusion & Refinement ---
        z_concat = torch.cat([z_cnn, z_seq], dim=1)  # [B, 192]
        z_fused = self.sgf(z_concat)
        z_se = self.se_block(z_fused.unsqueeze(1))
        z_mha = self.mha_recal(z_se)
        z_final = self.dropout(z_mha.squeeze(1))
        
        # --- 4. Task-Specific Output ---
        out = self.head(z_final)
        
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return out.view(out.shape[0], self.pred_len, self.c_out)
        elif self.task_name == 'imputation':
            return out.view(out.shape[0], self.seq_len, self.c_out)
        else:  # classification
            return out
