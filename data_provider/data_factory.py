import os
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from aeon.datasets import load_classification


# ============================================================================
# 1. Dataset for PEMS Forecasting Tasks
# ============================================================================
class Dataset_PEMS(Dataset):

    def __init__(self,
                 root_path,
                 flag='train',
                 size=None,
                 features='M',
                 data_path='data.npz',
                 scale=True):

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.root_path = root_path
        self.data_path = data_path

        self.scale = scale
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):

        data = np.load(
            os.path.join(self.root_path, self.data_path)
        )['data']

        # Use traffic flow feature
        if data.ndim == 3:
            data = data[:, :, 0]

        data = np.nan_to_num(data, nan=0.0).astype(np.float32)

        total_len = len(data)

        # ============================================================
        # 60 / 20 / 20 split
        # ============================================================

        train_end = int(total_len * 0.6)
        val_end = int(total_len * 0.8)

        border1s = [
            0,
            train_end - self.seq_len,
            val_end - self.seq_len
        ]

        border2s = [
            train_end,
            val_end,
            total_len
        ]

        type_map = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        set_type = type_map[self.flag]

        if self.flag == 'train':
            b1, b2 = border1s[set_type], border2s[set_type]
        else:
            b1 = border1s[set_type] - self.seq_len
            b2 = border2s[set_type]

        # ============================================================
        # Scaling
        # ============================================================

        if self.scale:

            train_data = data[
                border1s[0]:border2s[0]
            ]

            self.scaler = StandardScaler()

            self.scaler.fit(train_data)

            data = self.scaler.transform(data).astype(np.float32)

        self.data_x = data[b1:b2]

    def __getitem__(self, index):

        s_begin = index
        s_end = s_begin + self.seq_len

        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_x[r_begin:r_end]

        return (
            torch.FloatTensor(seq_x),
            torch.FloatTensor(seq_y)
        )

    def __len__(self):

        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):

        if self.scale:
            return self.scaler.inverse_transform(data)

        return data


# ============================================================================
# 2. Dataset for Imputation Tasks
# ============================================================================
class Dataset_Imputation(Dataset):

    def __init__(self,
                 root_path,
                 data_path,
                 flag='train',
                 seq_len=96,
                 mask_ratio=0.25,
                 target='OT',
                 scale=True):

        self.seq_len = seq_len
        self.mask_ratio = mask_ratio

        self.root_path = root_path
        self.data_path = data_path

        self.target = target
        self.scale = scale
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):

        df_raw = pd.read_csv(
            os.path.join(self.root_path, self.data_path)
        )

        # ============================================================
        # Official Splits
        # ============================================================

        # ETTh1 / ETTh2 -> 60 / 20 / 20
        if 'ETTh' in self.data_path:

            train_end = 8545
            val_end = train_end + 2881
            test_end = val_end + 2881

        # ETTm1 / ETTm2 -> 60 / 20 / 20
        elif 'ETTm' in self.data_path:

            train_end = 34465
            val_end = train_end + 11521
            test_end = val_end + 11521

        # Electricity -> 70 / 10 / 20
        elif 'electricity' in self.data_path.lower():

            train_end = 18317
            val_end = train_end + 2633
            test_end = val_end + 5261

        # Weather -> 70 / 10 / 20
        elif 'weather' in self.data_path.lower():

            train_end = 36792
            val_end = train_end + 5271
            test_end = val_end + 10540

        else:

            total_len = len(df_raw)

            train_end = int(total_len * 0.7)
            val_end = int(total_len * 0.8)
            test_end = total_len

        border1s = [
            0,
            train_end - self.seq_len,
            val_end - self.seq_len
        ]

        border2s = [
            train_end,
            val_end,
            test_end
        ]

        type_map = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        set_type = type_map[self.flag]

        if self.flag == 'train':
            b1, b2 = border1s[set_type], border2s[set_type]
        else:
            b1 = border1s[set_type] - self.seq_len
            b2 = border2s[set_type]

        # ============================================================
        # Remove datetime column
        # ============================================================

        if 'date' in df_raw.columns:
            df_data = df_raw.drop(['date'], axis=1)
        else:
            df_data = df_raw

        data = df_data.values.astype(np.float32)

        # ============================================================
        # Scaling
        # ============================================================

        if self.scale:

            train_data = data[
                border1s[0]:border2s[0]
            ]

            self.scaler = StandardScaler()

            self.scaler.fit(train_data)

            data = self.scaler.transform(data).astype(np.float32)

        self.data = data[b1:b2]

    def __getitem__(self, index):

        s_begin = index
        s_end = s_begin + self.seq_len

        seq_original = self.data[s_begin:s_end]

        # Random Mask
        mask = (
            torch.rand(seq_original.shape)
            > self.mask_ratio
        )

        seq_masked = seq_original * mask.numpy()

        return (
            torch.FloatTensor(seq_masked),
            torch.FloatTensor(seq_original),
            mask.float()
        )

    def __len__(self):

        return len(self.data) - self.seq_len + 1


# ============================================================================
# 3. Dataset for UEA Classification Tasks
# ============================================================================
class Dataset_UEA(Dataset):

    def __init__(self,
                 dataset_name,
                 flag='train',
                 label_encoder=None,
                 norm_stats=None):

        self.dataset_name = dataset_name
        self.flag = flag

        self.label_encoder = label_encoder
        self.norm_stats = norm_stats

        self.__read_data__()

    def __read_data__(self):

        split = "TRAIN" if self.flag == "train" else "TEST"

        X, y = load_classification(
            self.dataset_name,
            split=split
        )

        # ============================================================
        # Padding variable-length sequences
        # ============================================================

        if isinstance(X, list):

            max_len = max(x.shape[1] for x in X)

            padded_X = []

            for x in X:

                if x.shape[1] < max_len:

                    pad_width = (
                        (0, 0),
                        (0, max_len - x.shape[1])
                    )

                    x = np.pad(
                        x,
                        pad_width,
                        mode='constant',
                        constant_values=0
                    )

                padded_X.append(x)

            self.data = np.stack(padded_X).astype(np.float32)

        else:

            self.data = X.astype(np.float32)

        # ============================================================
        # Label Encoding
        # ============================================================

        if self.flag == 'train':

            self.label_encoder = LabelEncoder()

            self.labels = self.label_encoder.fit_transform(y)

        else:

            self.labels = self.label_encoder.transform(y)

        # ============================================================
        # Channel-wise normalization
        # ============================================================

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

                    self.data[:, c, :] = (
                        (channel_data - mean) / std
                    )

        else:

            self.channel_means, self.channel_stds = self.norm_stats

            for c in range(self.data.shape[1]):

                mean = self.channel_means[c]
                std = self.channel_stds[c]

                if std > 1e-8:

                    self.data[:, c, :] = (
                        (self.data[:, c, :] - mean) / std
                    )

    def __getitem__(self, index):

        return (
            torch.FloatTensor(self.data[index]),
            torch.LongTensor([self.labels[index]])[0]
        )

    def __len__(self):

        return len(self.data)


# ============================================================================
# 4. Dataset for Cross-Domain Evaluation
# ============================================================================
class Dataset_CrossDomain(Dataset):

    def __init__(self,
                 root_path,
                 flag='train',
                 size=None,
                 features='M',
                 data_path='data.csv',
                 target='label',
                 scale=True):

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.features = features
        self.target = target

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path

        assert flag in ['train', 'val', 'test']

        type_map = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        self.set_type = type_map[flag]

        self.__read_data__()

    def __read_data__(self):

        df_raw = pd.read_csv(
            os.path.join(self.root_path, self.data_path)
        )

        total_len = len(df_raw)

        # ============================================================
        # 80 / 10 / 10 Split
        # ============================================================

        num_train = int(total_len * 0.8)
        num_vali = int(total_len * 0.1)
        num_test = total_len - num_train - num_vali

        border1s = [
            0,
            num_train - self.seq_len,
            total_len - num_test - self.seq_len
        ]

        border2s = [
            num_train,
            num_train + num_vali,
            total_len
        ]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # ============================================================
        # Feature Selection
        # ============================================================

        if self.features in ['M', 'MS']:

            cols_data = [
                col for col in df_raw.columns
                if col != self.target
            ]

            df_data = df_raw[cols_data]
            df_labels = df_raw[[self.target]]

        else:

            df_data = df_raw[[self.target]]
            df_labels = df_raw[[self.target]]

        data = df_data.values.astype(np.float32)

        # ============================================================
        # Scaling
        # ============================================================

        if self.scale:

            train_data = data[
                border1s[0]:border2s[0]
            ]

            self.scaler = StandardScaler()

            self.scaler.fit(train_data)

            data = self.scaler.transform(data).astype(np.float32)

        labels = df_labels.values.astype(np.int64)

        self.data_x = data[border1:border2]
        self.data_y = labels[border1:border2]

    def __getitem__(self, index):

        s_begin = index
        s_end = s_begin + self.seq_len

        seq_x = self.data_x[s_begin:s_end]

        if len(self.data_y.shape) == 1:
            label = self.data_y[s_end - 1]
        else:
            label = self.data_y[s_end - 1, 0]

        return (
            torch.FloatTensor(seq_x),
            torch.LongTensor([int(label)])[0]
        )

    def __len__(self):

        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):

        if self.scale:
            return self.scaler.inverse_transform(data)

        return data


# ============================================================================
# DATA PROVIDER
# ============================================================================
def data_provider(args, flag):

    shuffle_flag = (flag == 'train')
    drop_last = (flag == 'train')

    batch_size = args.batch_size

    # ============================================================
    # Forecasting
    # ============================================================

    if args.task_name == 'forecasting':

        if args.dataset_name in [
            'PEMS03',
            'PEMS04',
            'PEMS07',
            'PEMS08'
        ]:

            data_set = Dataset_PEMS(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[
                    args.seq_len,
                    args.label_len,
                    args.pred_len
                ],
                features='M',
                scale=True
            )

        else:

            raise ValueError(
                f'Unsupported forecasting dataset: {args.dataset_name}'
            )

    # ============================================================
    # Imputation
    # ============================================================

    elif args.task_name == 'imputation':

        if args.dataset_name in [
            'ETTh1',
            'ETTh2',
            'ETTm1',
            'ETTm2',
            'ECL',
            'Weather'
        ]:

            data_set = Dataset_Imputation(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                seq_len=args.seq_len,
                mask_ratio=args.mask_ratio,
                target=args.target,
                scale=True
            )

        else:

            raise ValueError(
                f'Unsupported imputation dataset: {args.dataset_name}'
            )

    # ============================================================
    # Classification
    # ============================================================

    elif args.task_name == 'classification':

        # --------------------------------------------------------
        # UEA datasets
        # --------------------------------------------------------

        if args.dataset_name in [
            'EthanolConcentration',
            'FaceDetection',
            'Handwriting',
            'Heartbeat',
            'JapaneseVowels',
            'PEMS-SF',
            'SelfRegulationSCP1',
            'SelfRegulationSCP2',
            'SpokenArabicDigits',
            'UWaveGestureLibrary'
        ]:

            if flag == 'train':

                data_set = Dataset_UEA(
                    dataset_name=args.dataset_name,
                    flag='train'
                )

                # Save normalization + label encoder
                args.label_encoder = data_set.label_encoder

                args.norm_stats = (
                    data_set.channel_means,
                    data_set.channel_stds
                )

            else:

                data_set = Dataset_UEA(
                    dataset_name=args.dataset_name,
                    flag=flag,
                    label_encoder=args.label_encoder,
                    norm_stats=args.norm_stats
                )

        # --------------------------------------------------------
        # Cross-domain datasets
        # --------------------------------------------------------

        elif args.dataset_name in [
            'PAMAP2',
            'UCI-HAR',
            'Rodegast',
            'Boubezoul',
            'ActBeCalf',
            'MetroPT3',
            'NASA'
        ]:

            data_set = Dataset_CrossDomain(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[
                    args.seq_len,
                    args.label_len,
                    args.pred_len
                ],
                features='M',
                target='label',
                scale=True
            )

        else:

            raise ValueError(
                f'Unsupported classification dataset: {args.dataset_name}'
            )

    else:

        raise ValueError(
            f'Unsupported task: {args.task_name}'
        )

    # ============================================================
    # DataLoader
    # ============================================================

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    return data_set, data_loader
