import torch
import torch.nn as nn
from layers.mstn_modules import MultiScaleCNN, BiLSTMPathway 
from layers.eta_module import ETA_Module
from layers.sgf_module import SGF_Module
from layers.se_block import SE_Block
from layers.single_dense_layer import SingleDenseLayer  

class MSTN_BiLSTM(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        
        # --- 1. Dual Pathways (Feature Extraction) ---
        self.cnn_pathway = MultiScaleCNN(configs.enc_in, cnn_hidden=64)
        self.seq_pathway = BiLSTMPathway(
            c_in=configs.enc_in,
            hidden_dim=64,  # Changed from lstm_hidden=128
            num_layers=2
        )
        
        # --- 2. Early Temporal Aggregation (ETA) ---
        self.eta_cnn = ETA_Module(pool_type='gap')    
        self.eta_seq = ETA_Module(pool_type='mean')  
        
        # --- 3. Fusion & Refinement ---
        fused_dim = 64 + 128  # 192
        self.sgf = SGF_Module(fused_dim)
        self.se_block = SE_Block(fused_dim, reduction=8)
        self.single_dense = SingleDenseLayer(fused_dim, dropout=configs.dropout)  # Changed
        self.dropout = nn.Dropout(configs.dropout)
        
        # --- 4. Task-Specific Output Heads ---
        if self.task_name == 'classification':
            self.head = nn.Linear(fused_dim, configs.num_class)
            
        elif self.task_name in ['long_term_forecast', 'cross_dataset_generalization']:
            self.pred_len = configs.pred_len
            self.c_out = configs.c_out
            self.head = nn.Linear(fused_dim, self.pred_len * self.c_out)
            
        elif self.task_name == 'imputation':
            self.c_out = configs.c_out
            self.head = nn.Linear(fused_dim, self.seq_len * self.c_out)
            
        else:
            raise ValueError(f"Task '{self.task_name}' is not supported by MSTN.")

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: [Batch, Length, Channels]

        # 1. Dual Pathway Processing
        h_cnn = self.cnn_pathway(x_enc.transpose(1, 2))  # [B, 64, L]
        h_seq = self.seq_pathway(x_enc)  # [B, L, 128]
        
        # 2. Aggregation (Collapses temporal dimension L -> 1)
        z_cnn = self.eta_cnn(h_cnn)  # [B, 64]
        z_seq = self.eta_seq(h_seq)  # [B, 128]
        
        # 3. Fusion & Refinement
        z_concat = torch.cat([z_cnn, z_seq], dim=1)  # [B, 192]
        z_fused = self.sgf(z_concat)
        
        # Refinement (requires 3D input [B, 1, C])
        z_se = self.se_block(z_fused.unsqueeze(1))
        z_dense = self.single_dense(z_se)  # Changed
        z_final = self.dropout(z_dense.squeeze(1))
        
        # 4. Task-Specific Output
        out = self.head(z_final)
        
        if self.task_name in ['long_term_forecast', 'cross_dataset_generalization']:
            return out.view(out.shape[0], self.pred_len, self.c_out)
        
        elif self.task_name == 'imputation':
            return out.view(out.shape[0], self.seq_len, self.c_out)
        
        return out
