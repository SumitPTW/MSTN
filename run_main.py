import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

from models import MSTN_Transformer, MSTN_BiLSTM
from data_provider.data_factory import data_provider


def train_epoch(model, train_loader, optimizer, criterion, task_name, device, pred_len=None):
    model.train()
    total_loss = 0
    
    for batch_data in train_loader:
        optimizer.zero_grad()
        
        if task_name == 'classification':
            x, y = batch_data
            x, y = x.to(device).float(), y.to(device).long()
            output = model(x)
            loss = criterion(output, y)
        
        elif task_name == 'imputation':
            x_masked, x_original, mask = batch_data
            x_masked = x_masked.to(device).float()
            x_original = x_original.to(device).float()
            mask = mask.to(device).float()
            output = model(x_masked)
            missing_mask = 1.0 - mask
            loss = criterion(output * missing_mask, x_original * missing_mask)

        elif task_name in ['long_term_forecast', 'cross_dataset_generalization']:
            if len(batch_data) == 2:
                x, y = batch_data
                x = x.to(device).float()
                y = y.to(device).float()
                output = model(x)
            else:
                x, y, x_mark, y_mark = batch_data
                x = x.to(device).float()
                y = y.to(device).float()
                x_mark = x_mark.to(device).float()
                y_mark = y_mark.to(device).float()
                output = model(x, x_mark, y, y_mark)
            
            if pred_len and y.shape[1] > pred_len:
                y = y[:, -pred_len:, :]
            
            loss = criterion(output, y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)


def evaluate(model, data_loader, task_name, device, pred_len=None):
    model.eval()
    predictions, targets = [], []
    
    with torch.no_grad():
        for batch_data in data_loader:
            if task_name == 'classification':
                x, y = batch_data
                output = model(x.to(device).float())
                pred = output.argmax(dim=1)
                predictions.extend(pred.cpu().numpy())
                targets.extend(y.numpy())
            
            elif task_name == 'imputation':
                x_masked, x_original, mask = batch_data
                output = model(x_masked.to(device).float())
                predictions.append(output.cpu().numpy())
                targets.append(x_original.numpy())
            
            elif task_name in ['long_term_forecast', 'cross_dataset_generalization']:
                if len(batch_data) == 2:
                    x, y = batch_data
                    output = model(x.to(device).float())
                    y = y.numpy()
                else:
                    x, y, x_mark, y_mark = batch_data
                    output = model(x.to(device).float(), 
                                   x_mark.to(device).float(), 
                                   y.to(device).float(), 
                                   y_mark.to(device).float())
                    y = y.numpy()
                
                if pred_len and y.shape[1] > pred_len:
                    y = y[:, -pred_len:, :]
                
                predictions.append(output.cpu().numpy())
                targets.append(y)
    
    if task_name == 'classification':
        return {'accuracy': accuracy_score(targets, predictions)}
    else:
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        mse = mean_squared_error(targets.flatten(), predictions.flatten())
        mae = mean_absolute_error(targets.flatten(), predictions.flatten())
        return {'MSE': mse, 'MAE': mae}


def main():
    parser = argparse.ArgumentParser(description='MSTN: Multi-Scale Temporal Network')
    
    parser.add_argument('--task_name', type=str, required=True, 
                       choices=['classification', 'imputation', 'long_term_forecast', 'cross_dataset_generalization'])
    parser.add_argument('--model', type=str, required=True, choices=['MSTN_Transformer', 'MSTN_BiLSTM'])
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--root_path', type=str, default='./datasets/')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--mask_ratio', type=float, default=0.25)
    parser.add_argument('--enc_in', type=int, default=None)
    parser.add_argument('--c_out', type=int, default=None)
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results/')
    
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.save_dir, exist_ok=True)
    experiment_name = f"{args.model}_{args.task_name}_{args.dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    save_path = os.path.join(args.save_dir, experiment_name)
    os.makedirs(save_path, exist_ok=True)
    
    if args.task_name in ['long_term_forecast', 'cross_dataset_generalization'] and not args.data_path:
        if args.dataset_name in ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']:
            args.data_path = f'{args.dataset_name}.npz'
        else:
            args.data_path = f'{args.dataset_name}.csv'
    
    train_dataset, train_loader = data_provider(args, flag='train')
    val_dataset, val_loader = data_provider(args, flag='val')
    test_dataset, test_loader = data_provider(args, flag='test')
    
    if args.enc_in is None and hasattr(train_dataset, 'data_x'):
        if len(train_dataset.data_x.shape) == 3:
            args.enc_in = train_dataset.data_x.shape[-1]
        else:
            args.enc_in = train_dataset.data_x.shape[-1]
    
    if args.c_out is None:
        args.c_out = args.enc_in
    
    model_dict = {'MSTN_Transformer': MSTN_Transformer, 'MSTN_BiLSTM': MSTN_BiLSTM}
    model = model_dict[args.model](args).to(device)
    
    print(f'Model: {args.model}')
    print(f'Task: {args.task_name}')
    print(f'Dataset: {args.dataset_name}')
    print(f'Enc_in: {args.enc_in}, C_out: {args.c_out}')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    criterion = nn.CrossEntropyLoss() if args.task_name == 'classification' else nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_metric = 0 if args.task_name == 'classification' else float('inf')
    history = {'train_loss': [], 'val_metric': []}
    
    for epoch in range(args.train_epochs):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.task_name, device, args.pred_len)
        val_results = evaluate(model, val_loader, args.task_name, device, args.pred_len)
        
        if args.task_name == 'classification':
            val_metric = val_results['accuracy']
            metric_name = 'Accuracy'
            better = val_metric > best_val_metric
        else:
            val_metric = val_results['MSE']
            metric_name = 'MSE'
            better = val_metric < best_val_metric
        
        history['train_loss'].append(train_loss)
        history['val_metric'].append(val_metric)
        
        if better:
            best_val_metric = val_metric
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            print(f'  New best {metric_name}: {val_metric:.4f}')
        
        scheduler.step(val_metric)
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1:03d}/{args.train_epochs} | Train Loss: {train_loss:.4f} | Val {metric_name}: {val_metric:.4f} | Time: {epoch_time:.1f}s')
    
    model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pth')))
    test_results = evaluate(model, test_loader, args.task_name, device, args.pred_len)
    
    pd.DataFrame(history).to_csv(os.path.join(save_path, 'training_history.csv'), index=False)
    pd.DataFrame([test_results]).to_csv(os.path.join(save_path, 'test_results.csv'), index=False)
    pd.DataFrame([vars(args)]).to_csv(os.path.join(save_path, 'config.csv'), index=False)
    
    print(f'\n{"="*50}')
    print(f'Test Results:')
    for k, v in test_results.items():
        print(f'  {k}: {v:.6f}')
    print(f'{"="*50}')
    print(f'Results saved to: {save_path}')


if __name__ == "__main__":
    main()
