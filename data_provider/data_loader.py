import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features

# ============================================================================
# 1. Dataset for PEMS Forecasting Tasks 
# ============================================================================
class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='data.npz', scale=True):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.flag = flag
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data = np.load(os.path.join(self.root_path, self.data_path))['data']
        
        # Use traffic flow data (feature index 0)
        if data.ndim == 3:
            data = data[:, :, 0]
        
        data = np.nan_to_num(data, nan=0.0)
        total_len = len(data)
        
        train_end = int(total_len * 0.6)
        val_end = int(total_len * 0.8)
        
        border1s = [0, train_end - self.seq_len, val_end - self.seq_len]
        border2s = [train_end, val_end, total_len]
        
        type_map = {'train': 0, 'val': 1, 'test': 2}
        set_type = type_map[self.flag]
        
        if self.flag == 'train':
            b1, b2 = border1s[set_type], border2s[set_type]
        else:
            b1, b2 = border1s[set_type] - self.seq_len, border2s[set_type]

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.reshape(-1, 1))
            data = self.scaler.transform(data.reshape(-1, 1)).reshape(-1)
        else:
            data = data

        self.data_x = data[b1:b2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_x[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data.reshape(-1, 1)).reshape(-1)


# ============================================================================
# 2. Dataset for Imputation Tasks (ETTh1, ETTh2, ETTm1, ETTm2, ECL, Weather)
# ============================================================================
class Dataset_Imputation(Dataset):
    def __init__(self, root_path, data_path, flag='train', seq_len=96,
                 mask_ratio=0.25, target='OT', scale=True):
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        self.root_path = root_path
        self.data_path = data_path
        self.target = target
        self.scale = scale
        self.flag = flag
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # Handle ETT datasets (60/20/20 split)
        if 'ETTh' in self.data_path or 'ETTm' in self.data_path:
            if 'ETTh' in self.data_path:
                train_end = 8640
                val_end = 11520
                test_end = 14400
            else:  # ETTm
                train_end = 34560
                val_end = 46080
                test_end = 57600
            
            border1s = [0, train_end - self.seq_len, val_end - self.seq_len]
            border2s = [train_end, val_end, test_end]
        else:
            # ECL, Weather (70/10/20 split)
            total_len = len(df_raw)
            train_end = int(total_len * 0.70)
            val_end = int(total_len * 0.80)
            
            border1s = [0, train_end - self.seq_len, val_end - self.seq_len]
            border2s = [train_end, val_end, total_len]
        
        type_map = {'train': 0, 'val': 1, 'test': 2}
        set_type = type_map[self.flag]
        
        if self.flag == 'train':
            b1, b2 = border1s[set_type], border2s[set_type]
        else:
            b1, b2 = border1s[set_type] - self.seq_len, border2s[set_type]

        if 'date' in df_raw.columns:
            df_data = df_raw.drop(['date'], axis=1)
        else:
            df_data = df_raw
            
        if self.scale:
            train_data = df_data.values[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            self.data = self.scaler.transform(df_data.values)
        else:
            self.data = df_data.values
        
        self.data = self.data[b1:b2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_original = self.data[s_begin:s_end]
        
        mask = torch.rand(seq_original.shape) > self.mask_ratio
        seq_masked = seq_original * mask.numpy()
        
        seq_original = torch.FloatTensor(seq_original)
        seq_masked = torch.FloatTensor(seq_masked)
        mask = mask.float()
        
        return seq_masked, seq_original, mask

    def __len__(self):
        return len(self.data) - self.seq_len + 1

# ============================================================================
# 3. Dataset for UEA Classification Tasks (10 datasets)
# ============================================================================
class Dataset_UEA(Dataset):
    """
    UEA classification dataset loader.
    Uses official UEA train/test splits from local .ts files.
    
    Output format: [channels, time_steps] matching your CNN input.
    """
    def __init__(self, dataset_name, flag='train', data_path='./datasets/UEA/'):
        """
        Args:
            dataset_name: Name of the UEA dataset (e.g., 'EthanolConcentration')
            flag: 'train' or 'test' (uses official UEA splits)
            data_path: Root path containing UEA dataset folders
        """
        self.dataset_name = dataset_name
        self.flag = flag
        self.data_path = data_path
        self.__read_data__()
    
    def __read_ts_file(self, file_path):
        """Read UEA .ts file format."""
        data, labels = [], []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split label and time series data
                parts = line.split(':')
                if len(parts) < 2:
                    continue
                
                # Extract label (first part)
                label = int(float(parts[0].strip()))
                labels.append(label)
                
                # Extract time series data (remaining parts)
                series_data = []
                channels_str = ':'.join(parts[1:])
                channels = channels_str.split('\t')
                
                for channel_str in channels:
                    if channel_str.strip():
                        values = [float(x.strip()) for x in channel_str.split(',') if x.strip()]
                        if values:
                            series_data.append(values)
                
                # Trim all channels to same length
                if series_data:
                    min_len = min(len(channel) for channel in series_data)
                    trimmed_data = [channel[:min_len] for channel in series_data]
                    # Shape: [time_steps, channels] → Transpose to [channels, time_steps]
                    data.append(np.array(trimmed_data).T)
        
        return data, labels
    
    def __read_data__(self):
        """Load train or test split from UEA archive."""
        train_file = os.path.join(self.data_path, self.dataset_name, f'{self.dataset_name}_TRAIN.ts')
        test_file = os.path.join(self.data_path, self.dataset_name, f'{self.dataset_name}_TEST.ts')
        
        if self.flag == 'train':
            file_to_load = train_file
        else:
            file_to_load = test_file
        
        if not os.path.exists(file_to_load):
            raise FileNotFoundError(f"UEA dataset file not found: {file_to_load}")
        
        data_list, labels_list = self.__read_ts_file(file_to_load)
        
        if not data_list:
            raise ValueError(f"No data loaded from {file_to_load}")
        
        # Pad sequences to same length
        max_len = max(seq.shape[1] for seq in data_list)  # time dimension
        n_channels = data_list[0].shape[0]  # channels
        
        padded_data = []
        for seq in data_list:
            curr_len = seq.shape[1]
            if curr_len < max_len:
                # Pad with zeros
                pad_width = ((0, 0), (0, max_len - curr_len))
                padded_seq = np.pad(seq, pad_width, mode='constant', constant_values=0)
            else:
                padded_seq = seq
            padded_data.append(padded_seq)
        
        self.data = np.stack(padded_data)  # Shape: [samples, channels, time_steps]
        self.labels = np.array(labels_list)
        
        # Z-score normalization per channel
        for channel_idx in range(self.data.shape[1]):
            channel_data = self.data[:, channel_idx, :]
            mean, std = channel_data.mean(), channel_data.std()
            if std > 1e-8:
                self.data[:, channel_idx, :] = (channel_data - mean) / std
    
    def __getitem__(self, index):
        """Returns: sequence [channels, time_steps], label"""
        sequence = torch.FloatTensor(self.data[index])  # [channels, time_steps]
        label = torch.LongTensor([self.labels[index]])[0]
        return sequence, label
    
    def __len__(self):
        return len(self.data) 
# ============================================================================
# 4. Dataset for Cross-Domain Evaluation (7 datasets)
# ============================================================================
class Dataset_CrossDomain(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='data.csv',
                 target='label', scale=True):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = [col for col in df_raw.columns if col != self.target]
            df_data = df_raw[cols_data]
            df_labels = df_raw[[self.target]]
        else:
            df_data = df_raw[[self.target]]
            df_labels = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        labels = df_labels.values

        self.data_x = data[border1:border2]
        self.data_y = labels[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        
        seq_x = self.data_x[s_begin:s_end]
        if len(self.data_y.shape) == 1:
            label = self.data_y[s_end-1]
        else:
            label = self.data_y[s_end-1, 0]
        
        return seq_x, label

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
