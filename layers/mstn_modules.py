import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleCNN(nn.Module):
    """
    Multi-scale Convolutional Neural Network pathway.
    """
    def __init__(self, c_in, cnn_hidden=64):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, 128, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, cnn_hidden, kernel_size=5, padding=2)

    def forward(self, x):
     
        # Conv1D_7 + BatchNorm + ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        # Conv1D_5 + ReLU
        x = F.relu(self.conv2(x))
        return x

class BiLSTMPathway(nn.Module):
    """
    Bidirectional LSTM pathway for sequence modeling.
       """
    def __init__(self, c_in, lstm_hidden=128, num_layers=2):
        super().__init__()
        # BiLSTM: hidden_size is per direction
        self.lstm = nn.LSTM(
            input_size=c_in,
            hidden_size=lstm_hidden // 2,  
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_hidden = lstm_hidden

    def forward(self, x):
               h_lstm, _ = self.lstm(x)
        return h_lstm

class TransformerPathway(nn.Module):
    """
    Transformer encoder pathway for sequence modeling.
    """
    def __init__(self, c_in, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        # Project input to transformer dimension
        self.input_proj = nn.Linear(c_in, d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
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
