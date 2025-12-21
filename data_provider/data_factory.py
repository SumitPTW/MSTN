
import torch
from torch.utils.data import DataLoader
from .data_loader import Dataset_Custom, Dataset_Imputation, Dataset_UEA, Dataset_HAR

def data_provider(args, flag):
   
    # Common DataLoader settings
    shuffle_flag = (flag == 'train')
    drop_last = (flag == 'train')
    batch_size = args.batch_size
    
    # --- ROUTE BASED ON TASK AND DATASET ---
    if args.task_name == 'forecasting':
        # Forecasting datasets (ETT, Electricity, Weather, etc.)
        if args.dataset_name in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 
                                'ECL', 'Weather', 'Traffic', 'Exchange', 'ILI']:
            data_set = Dataset_Custom(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                target=args.target,
                timeenc=args.embed == 'timeF',
                freq=args.freq
            )
        else:
            raise ValueError(f"Forecasting dataset {args.dataset_name} not supported.")
    
    elif args.task_name == 'imputation':
        # Imputation datasets (with masking) - Table 1b in paper
        if args.dataset_name in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ECL', 'Weather']:
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
            raise ValueError(f"Imputation dataset {args.dataset_name} not supported.")
    
    elif args.task_name == 'classification':
        # Classification datasets - Table 3a (UEA) and Table 3b (Cross-domain)
        if args.dataset_name in [
            'EthanolConcentration', 'FaceDetection', 'Handwriting', 'Heartbeat',
            'JapaneseVowels', 'PEMS-SF', 'SelfRegulationSCP1', 'SelfRegulationSCP2',
            'SpokenArabicDigits', 'UWaveGestureLibrary'
        ]:
            # UEA Archive datasets
            data_set = Dataset_UEA(
                dataset_name=args.dataset_name,
                flag=flag,
                data_path=args.root_path  
            )
        
        elif args.dataset_name in [
            'PAMAP2', 'UCI-HAR', 'Rodegast', 'Boubezoul',
            'ActBeCalf', 'MetroPT3', 'NASA'
        ]:
            # Cross-domain datasets (Table 3b)
            data_set = Dataset_HAR(
                dataset_name=args.dataset_name,
                flag=flag,
                data_path=args.root_path  # Assuming HAR data is in root_path
            )
        
        else:
            raise ValueError(f"Classification dataset {args.dataset_name} not supported.")
    
    else:
        raise ValueError(f"Unsupported task: {args.task_name}. Use 'forecasting', 'imputation', or 'classification'.")
    
    # Create DataLoader
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    
    return data_set, data_loader
