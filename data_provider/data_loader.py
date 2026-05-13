import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from aeon.datasets import load_classification


# ============================================================================
# 1. Dataset for PEMS Forecasting Tasks
# ============================================================================
class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='data.npz', scale=True):
        self.seq_len = size[0]
        self.pred_len = size[2]
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.flag = flag
        self.__read_data__()

    def __read_data__(self):
        data = np.load(os.path.join(self.root_path, self.data_path))['data']
        
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
            self.scaler = StandardScaler()
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)
            data = data.astype(np.float32)
        
        self.data_x = data[b1:b2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_x[r_begin:r_end]
        
        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        if self.scale:
            return self.scaler.inverse_transform(data)
        return data


# ============================================================================
# 2. Dataset for Imputation Tasks (ETTh, ETTm, Electricity, Weather)
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
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        if 'ETTh' in self.data_path:
            train_end = 8545
            val_end = train_end + 2881
            test_end = val_end + 2881
        elif 'ETTm' in self.data_path:
            train_end = 34465
            val_end = train_end + 11521
            test_end = val_end + 11521
        elif 'electricity' in self.data_path.lower():
            train_end = 18317
            val_end = train_end + 2633
            test_end = val_end + 5261
        elif 'weather' in self.data_path.lower():
            train_end = 36792
            val_end = train_end + 5271
            test_end = val_end + 10540
        else:
            total_len = len(df_raw)
            train_end = int(total_len * 0.70)
            val_end = int(total_len * 0.80)
            test_end = total_len
        
        border1s = [0, train_end - self.seq_len, val_end - self.seq_len]
        border2s = [train_end, val_end, test_end]
        
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
            self.scaler = StandardScaler()
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data.values)
            data = data.astype(np.float32)
        else:
            data = df_data.values.astype(np.float32)
        
        self.data = data[b1:b2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_original = self.data[s_begin:s_end]
        
        mask = torch.rand(seq_original.shape) > self.mask_ratio
        seq_masked = seq_original * mask.numpy()
        
        return (torch.FloatTensor(seq_masked), 
                torch.FloatTensor(seq_original), 
                mask.float())

    def __len__(self):
        return len(self.data) - self.seq_len + 1


# ============================================================================
# 3. Dataset for UEA Classification Tasks (10 Datasets)
# ============================================================================
class Dataset_UEA(Dataset):
    def __init__(self, dataset_name, flag='train', label_encoder=None, 
                 norm_stats=None, data_path='./datasets/UEA/'):
        self.dataset_name = dataset_name
        self.flag = flag
        self.data_path = data_path
        self.label_encoder = label_encoder
        self.norm_stats = norm_stats
        self.__read_data__()
    
    def __read_data__(self):
        split = "TRAIN" if self.flag == "train" else "TEST"
        X, y = load_classification(self.dataset_name, split=split)
        
        if isinstance(X, list):
            max_len = max(x.shape[1] for x in X)
            padded_X = []
            for x in X:
                if x.shape[1] < max_len:
                    pad_width = ((0, 0), (0, max_len - x.shape[1]))
                    padded = np.pad(x, pad_width, mode='constant', constant_values=0)
                else:
                    padded = x
                padded_X.append(padded)
            self.data = np.stack(padded_X).astype(np.float32)
        else:
            self.data = X.astype(np.float32)
        
        if self.flag == 'train':
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(y)
        else:
            self.labels = self.label_encoder.transform(y)
        
        if self.flag == 'train':
            self.channel_means = []
            self.channel_stds = []
            for c in range(self.data.shape[1]):
                channel_data = self.data[:, c, :]
                mean = channel_data.mean()
                std = channel_data.std()
                self.channel_means.append(mean)
                self.channel_stds.append(std)
                if std > 1e-8:
                    self.data[:, c, :] = (channel_data - mean) / std
        else:
            if self.norm_stats is not None:
                self.channel_means, self.channel_stds = self.norm_stats
                for c in range(self.data.shape[1]):
                    mean = self.channel_means[c]
                    std = self.channel_stds[c]
                    if std > 1e-8:
                        self.data[:, c, :] = (self.data[:, c, :] - mean) / std
    
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]), torch.LongTensor([self.labels[index]])[0]
    
    def __len__(self):
        return len(self.data)


# ============================================================================
# 4. Dataset for Cross-Domain Evaluation
# ============================================================================
class Dataset_CrossDomain(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='data.csv',
                 target='label', scale=True):
        self.seq_len = size[0]
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
            self.scaler = StandardScaler()
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            data = data.astype(np.float32)
        else:
            data = df_data.values.astype(np.float32)
        
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
        
        return torch.FloatTensor(seq_x), torch.LongTensor([int(label)])[0]

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        if self.scale:
            return self.scaler.inverse_transform(data)
        return data
