import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleCNN(nn.Module):
    """
    Multi-scale Convolutional Neural Network pathway.
    Input: [B, C_in, L]
    Output: [B, 64, L]
    """
    def __init__(self, c_in, cnn_hidden=64):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, 128, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, cnn_hidden, kernel_size=5, padding=2)

    def forward(self, x):
        # Conv1D 7 → 128 channels
        x = F.relu(self.bn1(self.conv1(x)))
        # Conv1D 5 → 64 channels
        x = F.relu(self.conv2(x))
        return x


class BiLSTMPathway(nn.Module):
    """
    Bidirectional LSTM pathway for sequence modeling.
    Input: [B, L, C_in]
    Output: [B, L, hidden_dim*2] where hidden_dim=64 → output 128
    """
    def __init__(self, c_in, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=c_in,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.total_dim = hidden_dim * 2  # 128

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        return h_lstm


class TransformerPathway(nn.Module):
    """
    Transformer encoder pathway for sequence modeling.
    Input: [B, L, C_in]
    Output: [B, L, d_model] where d_model=128
    """
    def __init__(self, c_in, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        # Project input to transformer dimension
        self.input_proj = nn.Linear(c_in, d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        x_proj = self.input_proj(x)
        return self.transformer(x_proj)
