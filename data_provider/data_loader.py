import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.timefeatures import time_features

# ============================================================================
# 1. Dataset for Forecasting Tasks
# ============================================================================
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


# ============================================================================
# 2. Dataset for Imputation Tasks
# ============================================================================
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
        seq_original = self.data[s_begin:s_end]  # [seq_len, channels]
      
        mask = torch.rand(seq_original.shape) > self.mask_ratio
        seq_masked = seq_original * mask.numpy()
        

        seq_original = torch.FloatTensor(seq_original)
        seq_masked = torch.FloatTensor(seq_masked)
        mask = mask.float()
        
        return seq_masked, seq_original, mask

    def __len__(self):
        return len(self.data) - self.seq_len + 1


# ============================================================================
# 3. Dataset for UEA Classification Tasks
# ============================================================================

class Dataset_UEA(Dataset):
    """
    Dataset for UEA multivariate time series classification.
    Loads .ts files from the UEA archive format.
    
    Used for: EthanolConcentration, FaceDetection, ..., UWaveGestureLibrary
    """
    def __init__(self, dataset_name, flag='train', data_path='./datasets/UEA/'):
        self.dataset_name = dataset_name
        self.flag = flag
        self.data_path = data_path
        self.__read_data__()
    
    def __read_ts_file(self, file_path):
        """Helper: Load a UEA .ts file into numpy arrays."""
        data, labels = [], []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                

                parts = line.split(':')
                if len(parts) < 2:
                    continue
                
                label = int(float(parts[0].strip()))
                labels.append(label)
                
    
                series_data = []
            
                channels_str = ':'.join(parts[1:])
                channels = channels_str.split('\\t')
                
                for channel_str in channels:
                    if channel_str.strip():
                    
                        values = [float(x.strip()) for x in channel_str.split(',') if x.strip()]
                        if values:
                            series_data.append(values)
                
            
                if series_data:
                    min_len = min(len(channel) for channel in series_data)
                    trimmed_data = [channel[:min_len] for channel in series_data]
                    # Transpose to shape [seq_len, n_channels]
                    data.append(np.array(trimmed_data).T)
        
        return data, labels
    
    def __read_data__(self):

        train_file = os.path.join(self.data_path, self.dataset_name, f'{self.dataset_name}_TRAIN.ts')
        test_file = os.path.join(self.data_path, self.dataset_name, f'{self.dataset_name}_TEST.ts')
        
   
        if self.flag == 'train':
            file_to_load = train_file
        else:  # 'test' or 'val'
            file_to_load = test_file
        
        if not os.path.exists(file_to_load):
            raise FileNotFoundError(f"UEA dataset file not found: {file_to_load}")
        
   
        data_list, labels_list = self.__read_ts_file(file_to_load)
        
        if not data_list:
            raise ValueError(f"No data loaded from {file_to_load}")
        
    
        max_len = max(seq.shape[0] for seq in data_list)
        n_channels = data_list[0].shape[1]
        
 
        padded_data = []
        for seq in data_list:
            pad_len = max_len - seq.shape[0]
            if pad_len > 0:
                padded_seq = np.pad(seq, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
            else:
                padded_seq = seq
            padded_data.append(padded_seq)
        
        self.data = np.stack(padded_data)  # [n_samples, seq_len, n_channels]
        self.labels = np.array(labels_list)
   
        for channel_idx in range(self.data.shape[2]):
            channel_data = self.data[:, :, channel_idx]
            mean, std = channel_data.mean(), channel_data.std()
            if std > 1e-8:  # Avoid division by zero
                self.data[:, :, channel_idx] = (channel_data - mean) / std
    
    def __getitem__(self, index):
        sequence = torch.FloatTensor(self.data[index])  # [seq_len, channels]
        label = torch.LongTensor([self.labels[index]])[0]  # scalar
        return sequence, label
    
    def __len__(self):
        return len(self.data)

# ============================================================================
# 4. Dataset for Cross-Domain 
# ============================================================================
class Dataset_CrossDomain(Dataset):
    def __init__(self, dataset_name, flag='train', data_path='./datasets/cross_domain/'):
        self.dataset_name = dataset_name
        self.flag = flag
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
  
        if self.dataset_name == 'Rodegast':
      
            csv_file = os.path.join(self.data_path, 'Rodegast', 'collision_data.csv')
            df = pd.read_csv(csv_file)
       
            all_data, all_labels = self._process_rodegest(df)
        
        elif self.dataset_name == 'Boubezoul':
      
            data_file = os.path.join(self.data_path, 'Boubezoul', 'fall_data.xlsx')
            df = pd.read_excel(data_file)
            all_data, all_labels = self._process_boubezoul(df)
        
        elif self.dataset_name == 'UCI-HAR':
      
            train_file = os.path.join(self.data_path, 'UCI-HAR', 'train', 'X_train.txt')
            train_labels = os.path.join(self.data_path, 'UCI-HAR', 'train', 'y_train.txt')
            all_data, all_labels = self._process_uci_har(train_file, train_labels)
        
        elif self.dataset_name == 'PAMAP2':
    
            data_file = os.path.join(self.data_path, 'PAMAP2', 'subject101.dat')
            all_data, all_labels = self._process_pamap2(data_file)
        
        elif self.dataset_name == 'ActBeCalf':
   
            csv_file = os.path.join(self.data_path, 'ActBeCalf', 'calf_behavior.csv')
            df = pd.read_csv(csv_file)
            all_data, all_labels = self._process_actbecalf(df)
        
        elif self.dataset_name == 'MetroPT3':

            csv_dir = os.path.join(self.data_path, 'MetroPT3', 'csv_files')
            all_data, all_labels = self._process_metropolitan(csv_dir)
        
        elif self.dataset_name == 'NASA':
    
            csv_file = os.path.join(self.data_path, 'NASA', 'train_FD001.txt')
            all_data, all_labels = self._process_nasa(csv_file)
        
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported.")
      
        n_samples = len(all_data)
        n_train = int(n_samples * 0.7)
        n_val = int(n_samples * 0.1)
        
        if self.flag == 'train':
            self.data = all_data[:n_train]
            self.labels = all_labels[:n_train]
        elif self.flag == 'val':
            self.data = all_data[n_train:n_train+n_val]
            self.labels = all_labels[n_train:n_train+n_val]
        else:  # test
            self.data = all_data[n_train+n_val:]
            self.labels = all_labels[n_train+n_val:]
        
        # Normalize per channel
        for i in range(self.data.shape[2]):
            channel_data = self.data[:, :, i]
            mean, std = channel_data.mean(), channel_data.std()
            if std > 0:
                self.data[:, :, i] = (channel_data - mean) / std

    def __getitem__(self, index):
        sequence = torch.FloatTensor(self.data[index])
        label = torch.LongTensor([self.labels[index]])[0]
        return sequence, label

    def __len__(self):
        return len(self.data)


    # ========== PROCESSING FUNCTIONS ==========
    
    def _process_rodegest(self, df):
        """Process Rodegast collision data from CSV"""
        # Extract sensor columns (adjust based on actual CSV structure)
        sensor_cols = [col for col in df.columns if 'sensor' in col.lower() 
                      or 'acc' in col.lower() or 'gyro' in col.lower()]
        
        sequences = []
        labels = []
        window_size = 100
        stride = 10
        
        for i in range(0, len(df) - window_size, stride):
            seq = df[sensor_cols].iloc[i:i+window_size].values  # [100, n_features]
            label = df['label'].iloc[i+window_size-1] if 'label' in df.columns else 0
            sequences.append(seq)
            labels.append(label)
        
        return np.array(sequences), np.array(labels)
    
    def _process_uci_har(self, data_file, label_file):
        """Process UCI-HAR from TXT files"""
        # Load from official UCI-HAR format
        data = np.loadtxt(data_file)  # Shape: [n_samples, 128*9=1152]
        labels = np.loadtxt(label_file, dtype=int)  # Shape: [n_samples]
        
        # Reshape to [n_samples, 128, 9]
        data = data.reshape(-1, 128, 9)
        
        # Convert labels from 1-6 to 0-5
        labels = labels - 1
        
        return data, labels
    
    def _process_pamap2(self, data_file):
        """Process PAMAP2 .dat files"""
        # PAMAP2 raw format: each line has timestamp, activityID, heart_rate, IMU data
        data = []
        labels = []
        
        with open(data_file, 'r') as f:
            for line in f:
                values = line.strip().split()
                if len(values) < 2:
                    continue
                
                # First value is timestamp, second is activity ID
                label = int(float(values[1]))
                
                # Remaining values are sensor readings (52 features)
                sensor_values = [float(v) for v in values[2:54]]  # Adjust based on actual format
                
                # You need to accumulate into windows (e.g., 100 timesteps)
                # This is simplified - you need proper windowing logic
                pass
        
        # Return as numpy arrays after windowing
        return np.array(data), np.array(labels)
    
    def _process_boubezoul(self, df):
        """Process Boubezoul fall detection from Excel"""
        # Extract features and labels
        feature_cols = [col for col in df.columns if col not in ['label', 'timestamp', 'subject']]
        label_col = 'label'
        
        sequences = []
        labels = []
        window_size = 50
        stride = 5
        
        for i in range(0, len(df) - window_size, stride):
            seq = df[feature_cols].iloc[i:i+window_size].values
            label = df[label_col].iloc[i+window_size-1] if label_col in df.columns else 0
            sequences.append(seq)
            labels.append(label)
        
        return np.array(sequences), np.array(labels)
    
    def _process_actbecalf(self, df):
        """Process ActBeCalf behavior data"""
        # Similar to Rodegast but with different column names
        feature_cols = [col for col in df.columns if 'sensor' in col or 'value' in col]
        label_col = 'behavior'
        
        sequences = []
        labels = []
        window_size = 200
        stride = 20
        
        for i in range(0, len(df) - window_size, stride):
            seq = df[feature_cols].iloc[i:i+window_size].values
            label = df[label_col].iloc[i+window_size-1] if label_col in df.columns else 0
            sequences.append(seq)
            labels.append(label)
        
        return np.array(sequences), np.array(labels)
    
    def _process_metropolitan(self, csv_dir):
        """Process MetroPT-3 CSV files"""
        all_data = []
        all_labels = []
        
        # MetroPT-3 has multiple CSV files for different sensors
        for csv_file in sorted(os.listdir(csv_dir)):
            if csv_file.endswith('.csv'):
                df = pd.read_csv(os.path.join(csv_dir, csv_file))
                
                # Extract features (adjust based on actual columns)
                feature_cols = [col for col in df.columns if 'value' in col or 'reading' in col]
                
                # Create sequences
                window_size = 100
                stride = 10
                
                for i in range(0, len(df) - window_size, stride):
                    seq = df[feature_cols].iloc[i:i+window_size].values
                    # MetroPT-3 is for anomaly detection - adjust label logic
                    label = 0  # Normal
                    if 'anomaly' in df.columns:
                        label = int(df['anomaly'].iloc[i+window_size-1] > 0)
                    
                    all_data.append(seq)
                    all_labels.append(label)
        
        return np.array(all_data), np.array(all_labels)
    
    def _process_nasa(self, csv_file):
        """Process NASA Turbofan degradation data"""
        # NASA format: each row is a time cycle, columns are sensor readings
        df = pd.read_csv(csv_file, sep=' ', header=None)
        
        # First columns: engine ID, cycle, operational settings
        # Remaining columns: sensor readings (typically 14-26 sensors)
        sensor_start_col = 3  # Adjust based on your dataset version
        
        sequences = []
        labels = []
        window_size = 50
        stride = 5
        
        # Group by engine ID
        engine_ids = df.iloc[:, 0].unique()
        
        for engine_id in engine_ids:
            engine_data = df[df.iloc[:, 0] == engine_id]
            sensor_data = engine_data.iloc[:, sensor_start_col:].values
            
            # Create sliding windows
            for i in range(0, len(sensor_data) - window_size, stride):
                seq = sensor_data[i:i+window_size, :]
                # Label: remaining useful life (RUL)
                rul = len(sensor_data) - (i + window_size)
                sequences.append(seq)
                labels.append(rul)
        
        return np.array(sequences), np.array(labels)
