import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleCNN(nn.Module):
    """CNN pathway for local temporal patterns. Output shape: [B, C, L] for pooling."""
    def __init__(self, c_in, cnn_hidden=64):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, 128, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, cnn_hidden, kernel_size=5, padding=2)

    def forward(self, x):
        # Input x shape: [Batch, Channels, Length]
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 128, L]
        x = F.relu(self.conv2(x))            # [B, cnn_hidden, L]
        return x  # Keep as [B, C, L] for channel-first pooling

class BiLSTMPathway(nn.Module):
    """BiLSTM pathway for long-range dependencies. Output shape: [B, L, H]."""
    def __init__(self, c_in, lstm_hidden=128, num_layers=2):
        super().__init__()
        # BiLSTM: hidden_size is per direction
        self.lstm = nn.LSTM(
            input_size=c_in,
            hidden_size=lstm_hidden // 2,  # Each direction gets half
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_hidden = lstm_hidden

    def forward(self, x):
        # Input x shape: [Batch, Length, Channels]
        h_lstm, _ = self.lstm(x)  # [B, L, lstm_hidden]
        return h_lstm

class TransformerPathway(nn.Module):
    """Transformer pathway for long-range dependencies. Output shape: [B, L, H]."""
    def __init__(self, c_in, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(c_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # Input x shape: [Batch, Length, Channels]
        x_proj = self.input_proj(x)  # [B, L, d_model]
        return self.transformer(x_proj)
