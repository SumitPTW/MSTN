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

def train_epoch(model, train_loader, optimizer, criterion, task_name, device):
     model.train()
    total_loss = 0
    for batch_idx, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Prepare data based on task
        if task_name == 'classification':
            x, y = batch_data
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
        
        elif task_name == 'imputation':
            x_masked, x_original, mask = batch_data
            x_masked, x_original, mask = x_masked.to(device), x_original.to(device), mask.to(device)
            output = model(x_masked)
            # Masked MSE loss
            loss = criterion(output * mask, x_original * mask)

     
        elif task_name in ['long_term_forecast', 'cross_dataset_generalization']:
            x, y, x_mark, y_mark = batch_data
            x, y = x.to(device).float(), y.to(device).float()
            x_mark = x_mark.to(device).float()
            y_mark = y_mark.to(device).float()

            output = model(x, x_mark, y, y_mark)
            
            # SLICING FIX: This is required for Informer/TimesNet style data
            y = y[:, -args.pred_len:, :].to(device) 
            
            loss = criterion(output, y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'  Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(train_loader)

def evaluate(model, data_loader, task_name, device):

    model.eval()
    predictions, targets = [], []
    
    with torch.no_grad():
        for batch_data in data_loader:
            if task_name == 'classification':
                x, y = batch_data
                x = x.to(device)
                output = model(x)
                pred = output.argmax(dim=1)
                predictions.extend(pred.cpu().numpy())
                targets.extend(y.numpy())
            
            elif task_name == 'imputation':
                x_masked, x_original, mask = batch_data
                x_masked = x_masked.to(device)
                output = model(x_masked)
                predictions.extend(output.cpu().numpy())
                targets.extend(x_original.numpy())
            
            elif task_name in ['long_term_forecast', 'short_term_forecast']:
                x, y, x_mark, y_mark = batch_data
                x = x.to(device)
                output = model(x, x_mark, y, y_mark)
                predictions.extend(output.cpu().numpy())
                targets.extend(y.numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate metrics based on task
    if task_name == 'classification':
        acc = accuracy_score(targets, predictions)
        return {'accuracy': acc}
    else:  # forecasting or imputation
        mse = mean_squared_error(targets.flatten(), predictions.flatten())
        mae = mean_absolute_error(targets.flatten(), predictions.flatten())
        return {'MSE': mse, 'MAE': mae}

def main():
    parser = argparse.ArgumentParser(description='MSTN: Multi-Scale Temporal Network')
    
    # ===== Required Arguments =====
       parser.add_argument('--task_name', type=str, required=True, 
                    choices=['classification', 'imputation', 'long_term_forecast', 'cross_dataset_generalization'],
                    help='Task to perform')
    
    parser.add_argument('--model', type=str, required=True,
                       choices=['MSTN_Transformer', 'MSTN_BiLSTM'],
                       help='Model variant')
    parser.add_argument('--dataset_name', type=str, required=True,
                       help='Dataset name (e.g., ETTh1, PAMAP2, EthanolConcentration)')
    
    # ===== Data Arguments =====
    parser.add_argument('--root_path', type=str, default='./datasets/',
                       help='Root path of datasets')
    parser.add_argument('--data_path', type=str, default='',
                       help='Data file path (for forecasting/imputation)')
    parser.add_argument('--seq_len', type=int, default=96,
                       help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=96,
                       help='Prediction length (for forecasting)')
    parser.add_argument('--mask_ratio', type=float, default=0.25,
                       help='Mask ratio for imputation (0.125, 0.25, 0.375, 0.5)')
    
    # ===== Model Arguments =====
    parser.add_argument('--enc_in', type=int, default=7,
                       help='Encoder input size')
    parser.add_argument('--c_out', type=int, default=7,
                       help='Output channels (for forecasting/imputation)')
    parser.add_argument('--num_class', type=int, default=10,
                       help='Number of classes (for classification)')
    parser.add_argument('--d_model', type=int, default=128,
                       help='Dimension of model')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=4,
                       help='Number of encoder layers')
    parser.add_argument('--d_ff', type=int, default=512,
                       help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # ===== Training Arguments =====
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--train_epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Data loader workers')
    
    # ===== Experimental Arguments =====
    parser.add_argument('--use_gpu', type=bool, default=True,
                       help='Use GPU')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--save_dir', type=str, default='./results/',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # ===== Device Setup =====
    device = torch.device(f'cuda:{args.gpu}' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # ===== Create Save Directory =====
    os.makedirs(args.save_dir, exist_ok=True)
    experiment_name = f"{args.model}_{args.task_name}_{args.dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    save_path = os.path.join(args.save_dir, experiment_name)
    os.makedirs(save_path, exist_ok=True)
    
    # ===== Data Loading =====
    print(f'\n{"="*60}')
    print(f'Loading data: {args.dataset_name} for {args.task_name}')
    print(f'{"="*60}')
    
    # For forecasting tasks, set default data_path if not provided
    if args.task_name in ['long_term_forecast', 'short_term_forecast'] and not args.data_path:
        args.data_path = f'{args.dataset_name}.csv'
    
    # Get data loaders
    train_dataset, train_loader = data_provider(args, flag='train')
    val_dataset, val_loader = data_provider(args, flag='val')
    test_dataset, test_loader = data_provider(args, flag='test')
    
    print(f'Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Test samples: {len(test_dataset)}')
    
    # ===== Model Initialization =====
    print(f'\nInitializing model: {args.model}')
    
    model_dict = {
        'MSTN_Transformer': MSTN_Transformer,
        'MSTN_BiLSTM': MSTN_BiLSTM,
    }
    
    model = model_dict[args.model](args).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # ===== Loss Function & Optimizer =====
    if args.task_name == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:  # forecasting or imputation
        criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs)
    
    # ===== Training Loop =====
    print(f'\n{"="*60}')
    print(f'Training started: {args.train_epochs} epochs')
    print(f'{"="*60}')
    
    best_val_metric = 0 if args.task_name == 'classification' else float('inf')
    history = {'train_loss': [], 'val_metric': []}
    
    for epoch in range(args.train_epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.task_name, device)
        history['train_loss'].append(train_loss)
        
        # Validate
        val_results = evaluate(model, val_loader, args.task_name, device)
        
        if args.task_name == 'classification':
            val_metric = val_results['accuracy']
            metric_name = 'Accuracy'
        else:
            val_metric = val_results['MSE']
            metric_name = 'MSE'
        
        history['val_metric'].append(val_metric)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if (args.task_name == 'classification' and val_metric > best_val_metric) or \
           (args.task_name != 'classification' and val_metric < best_val_metric):
            best_val_metric = val_metric
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            print(f'  ✓ New best model saved! ({metric_name}: {val_metric:.4f})')
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1:03d}/{args.train_epochs} | '
              f'Time: {epoch_time:.1f}s | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val {metric_name}: {val_metric:.4f} | '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')
    
    # ===== Final Evaluation =====
    print(f'\n{"="*60}')
    print('Final Evaluation on Test Set')
    print(f'{"="*60}')
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pth')))
    
    # Test evaluation
    test_results = evaluate(model, test_loader, args.task_name, device)
    
    # Print results
    print(f'\nTest Results for {args.dataset_name}:')
    for metric, value in test_results.items():
        print(f'  {metric}: {value:.4f}')
    
    # ===== Save Results =====
    results_df = pd.DataFrame({
        'epoch': list(range(1, args.train_epochs + 1)),
        'train_loss': history['train_loss'],
        'val_metric': history['val_metric']
    })
    
    results_df.to_csv(os.path.join(save_path, 'training_history.csv'), index=False)
    
    # Save test results
    test_results_df = pd.DataFrame([test_results])
    test_results_df.to_csv(os.path.join(save_path, 'test_results.csv'), index=False)
    
    # Save configuration
    config_df = pd.DataFrame([vars(args)])
    config_df.to_csv(os.path.join(save_path, 'config.csv'), index=False)
    
    print(f'\nResults saved to: {save_path}')
    print(f'\n{"="*60}')
    print('Experiment completed successfully!')
    print(f'{"="*60}')

if __name__ == "__main__":
    main()
