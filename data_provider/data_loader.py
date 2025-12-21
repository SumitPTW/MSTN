import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='data.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.root_path = root_path
        self.data_path = data_path
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_val = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df_raw)]
        
        type_map = {'train': 0, 'val': 1, 'test': 2}
        set_type = type_map[self.flag]
        b1, b2 = border1s[set_type], border2s[set_type]

        if 'date' in df_raw.columns:
            df_stamp = df_raw[['date']][b1:b2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(['date'], axis=1).values
            else:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
        else:
            # If no date (classification), create dummy marks
            data_stamp = torch.zeros((b2 - b1, 1))

        cols_data = [c for c in df_raw.columns if c not in ['date', self.target]]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data.values[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[b1:b2]
        self.data_y = df_raw[self.target].values[b1:b2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_x[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


class Dataset_Imputation(Dataset):
    """
    Dataset for time series imputation tasks.
    Applies random masking to simulate missing values.
    
    Used for: ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Weather
    with mask_ratio = [0.125, 0.25, 0.375, 0.5]
    """
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
        # Similar to Dataset_Custom but NO time features needed
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # Same train/val/test split logic
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_val = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df_raw)]
        
        type_map = {'train': 0, 'val': 1, 'test': 2}
        set_type = type_map[self.flag]
        b1, b2 = border1s[set_type], border2s[set_type]
        
        # Prepare data (without date columns)
        if 'date' in df_raw.columns:
            df_data = df_raw.drop(['date'], axis=1)
        else:
            df_data = df_raw
            
        # Scale if needed
        if self.scale:
            train_data = df_data.values[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            self.data = self.scaler.transform(df_data.values)
        else:
            self.data = df_data.values
        
        self.data = self.data[b1:b2]  # Slice for current set

    def __getitem__(self, index):
        # Extract sequence
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_original = self.data[s_begin:s_end]  # [seq_len, channels]
        
        # Create random mask
        mask = torch.rand(seq_original.shape) > self.mask_ratio
        seq_masked = seq_original * mask.numpy()
        
        # Convert to tensors
        seq_original = torch.FloatTensor(seq_original)
        seq_masked = torch.FloatTensor(seq_masked)
        mask = mask.float()
        
        # For imputation: return masked sequence, original sequence, and mask
        return seq_masked, seq_original, mask

    def __len__(self):
        return len(self.data) - self.seq_len + 1


class Dataset_UEA(Dataset):
    """
    Dataset for UEA multivariate time series classification.
    Loads .ts files from the UEA archive format.
    
    Used for: EthanolConcentration, FaceDetection, ..., UWaveGestureLibrary
    """
    def __init__(self, dataset_name, flag='train', data_path='path/to/UEA/archive/'):
        self.dataset_name = dataset_name
        self.flag = flag
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        # UEA datasets are in .ts format
        # You need to implement actual loading from .ts files
        train_file = os.path.join(self.data_path, f'{self.dataset_name}/{self.dataset_name}_TRAIN.ts')
        test_file = os.path.join(self.data_path, f'{self.dataset_name}/{self.dataset_name}_TEST.ts')
        
        # This is a SKELETON - you need to implement actual .ts loading
        # UEA .ts format: First column is label, rest are time series
        
        if self.flag == 'train':
            # Load training data
            self.data = ...  # Shape: [n_samples, seq_len, n_channels]
            self.labels = ...  # Shape: [n_samples]
        else:
            # Load test data
            self.data = ...
            self.labels = ...
        
        # Normalize per channel
        self.data = (self.data - self.data.mean(axis=0)) / (self.data.std(axis=0) + 1e-8)

    def __getitem__(self, index):
        # For classification: return sequence and label
        sequence = torch.FloatTensor(self.data[index])  # [seq_len, channels]
        label = torch.LongTensor([self.labels[index]])[0]  # scalar
        return sequence, label

    def __len__(self):
        return len(self.data)
