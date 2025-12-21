import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        
        self.conv1 = nn.Conv1d(configs.enc_in, 128, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        
        self.bilstm = nn.LSTM(
            input_size=configs.enc_in, hidden_size=64, 
            num_layers=2, batch_first=True, bidirectional=True
        )

        self.gate_weight = nn.Linear(192, 192)
        self.se_fc1 = nn.Linear(192, 24)
        self.se_fc2 = nn.Linear(24, 192)
        
        self.mha_recal = nn.MultiheadAttention(embed_dim=192, num_heads=4, batch_first=True)
        self.ln_final = nn.LayerNorm(192)
        self.dropout = nn.Dropout(0.3)

        if self.task_name == 'classification':
            self.head = nn.Linear(192, configs.num_class)
        else:
            self.head = nn.Linear(192, configs.c_out)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        h_conv2 = F.relu(self.conv2(F.relu(self.bn1(self.conv1(x_enc.transpose(1, 2))))))
        z_cnn = F.avg_pool1d(h_conv2, h_conv2.size(2)).squeeze(-1)

        h_bilstm, _ = self.bilstm(x_enc)
        z_bilstm = torch.mean(h_bilstm, dim=1)

        z_concat = torch.cat([z_cnn, z_bilstm], dim=-1)
        z_fused = z_concat * torch.sigmoid(self.gate_weight(z_concat))
        
        z_seq = z_fused.unsqueeze(1)
        z_se = z_seq * torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(z_seq))))
        z_mha, _ = self.mha_recal(z_se, z_se, z_se)
        z_final = self.dropout(self.ln_final(z_mha.squeeze(1)))

        return self.head(z_final)