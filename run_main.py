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
    for batch_idx, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        
        if task_name == 'classification':
            x, y = batch_data
            x, y = x.to(device).float(), y.to(device).long()
            output = model(x)
            loss = criterion(output, y)
        
        elif task_name == 'imputation':
            x_masked, x_original, mask = batch_data
            x_masked, x_original, mask = x_masked.to(device).float(), x_original.to(device).float(), mask.to(device).float()
            output = model(x_masked)
            loss = criterion(output * mask, x_original * mask)

        elif task_name in ['long_term_forecast', 'cross_dataset_generalization']:
            x, y, x_mark, y_mark = batch_data
            x, y = x.to(device).float(), y.to(device).float()
            x_mark, y_mark = x_mark.to(device).float(), y_mark.to(device).float()

            output = model(x, x_mark, y, y_mark)
            
            # SLICING FIX: Compare only to the prediction horizon
            if pred_len:
                y = y[:, -pred_len:, :].to(device)
            
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
            
            elif task_name in ['long_term_forecast', 'short_term_forecast', 'cross_dataset_generalization']:
                x, y, x_mark, y_mark = batch_data
                output = model(x.to(device).float(), x_mark.to(device).float(), y.to(device).float(), y_mark.to(device).float())
                
                # SLICING FIX: Must match train_epoch logic
                if pred_len:
                    y = y[:, -pred_len:, :]
                
                predictions.append(output.cpu().numpy())
                targets.append(y.cpu().numpy())
    
    if task_name == 'classification':
        return {'accuracy': accuracy_score(targets, predictions)}
    else:
        # Proper concatenation for 3D forecasting/imputation arrays
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        mse = mean_squared_error(targets.flatten(), predictions.flatten())
        mae = mean_absolute_error(targets.flatten(), predictions.flatten())
        return {'MSE': mse, 'MAE': mae}

def main():
    parser = argparse.ArgumentParser(description='MSTN: Multi-Scale Temporal Network')
    
    # Arguments
    parser.add_argument('--task_name', type=str, required=True, choices=['classification', 'imputation', 'long_term_forecast', 'cross_dataset_generalization'])
    parser.add_argument('--model', type=str, required=True, choices=['MSTN_Transformer', 'MSTN_BiLSTM'])
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--root_path', type=str, default='./datasets/')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--mask_ratio', type=float, default=0.25)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results/')
    
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    
    # Path Setup
    os.makedirs(args.save_dir, exist_ok=True)
    experiment_name = f"{args.model}_{args.task_name}_{args.dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    save_path = os.path.join(args.save_dir, experiment_name)
    os.makedirs(save_path, exist_ok=True)
    
    if args.task_name in ['long_term_forecast', 'short_term_forecast'] and not args.data_path:
        args.data_path = f'{args.dataset_name}.csv'
    
    # Data
    train_dataset, train_loader = data_provider(args, flag='train')
    val_dataset, val_loader = data_provider(args, flag='val')
    test_dataset, test_loader = data_provider(args, flag='test')
    
    # Model
    model_dict = {'MSTN_Transformer': MSTN_Transformer, 'MSTN_BiLSTM': MSTN_BiLSTM}
    model = model_dict[args.model](args).to(device)
    
    # Print Params
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Optims
    criterion = nn.CrossEntropyLoss() if args.task_name == 'classification' else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs)
    
    best_val_metric = 0 if args.task_name == 'classification' else float('inf')
    history = {'train_loss': [], 'val_metric': []}
    
    for epoch in range(args.train_epochs):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.task_name, device, args.pred_len)
        val_results = evaluate(model, val_loader, args.task_name, device, args.pred_len)
        
        val_metric = val_results['accuracy'] if args.task_name == 'classification' else val_results['MSE']
        metric_name = 'Accuracy' if args.task_name == 'classification' else 'MSE'
        
        history['train_loss'].append(train_loss)
        history['val_metric'].append(val_metric)
        
        # Save Best
        if (args.task_name == 'classification' and val_metric > best_val_metric) or \
           (args.task_name != 'classification' and val_metric < best_val_metric):
            best_val_metric = val_metric
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            print(f'  ✓ New best {metric_name}: {val_metric:.4f}')
        
        scheduler.step()
        print(f'Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val {metric_name}: {val_metric:.4f}')
    
    # Test
    model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pth')))
    test_results = evaluate(model, test_loader, args.task_name, device, args.pred_len)
    
    # Save Files
    pd.DataFrame(history).to_csv(os.path.join(save_path, 'training_history.csv'), index=False)
    pd.DataFrame([test_results]).to_csv(os.path.join(save_path, 'test_results.csv'), index=False)
    pd.DataFrame([vars(args)]).to_csv(os.path.join(save_path, 'config.csv'), index=False)
    print(f'Test Results: {test_results}')

if __name__ == "__main__":
    main()
