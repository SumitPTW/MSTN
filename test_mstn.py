#!/usr/bin/env python3
"""
MSTN Imputation M→M Test Script
Shows Multivariate to Multivariate for all datasets
"""
import sys
import torch

sys.path.append('.')

def test_imputation_m2m(model_name, dataset_name, num_features):
    """Test M→M imputation for a specific dataset."""
    
    print(f"\n{'─'*50}")
    print(f"Testing: {model_name} on {dataset_name}")
    print(f"{'─'*50}")
    
    try:
        if model_name == 'MSTN_Transformer':
            from models import MSTN_Transformer as ModelClass
        else:
            from models import MSTN_BiLSTM as ModelClass
        
        # Configuration for imputation
        class Config:
            task_name = 'imputation'
            seq_len = 96
            pred_len = 96
            enc_in = num_features
            c_out = num_features
            dropout = 0.1
            num_class = 10
            d_model = 128
            n_heads = 8
            e_layers = 4
            d_ff = 512
        
        config = Config()
        model = ModelClass(config)
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {params:,}")
        
        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, config.seq_len, config.enc_in)
        output = model(x)
        
        print(f"   Input:  {x.shape} ({num_features} features)")
        print(f"   Output: {output.shape} ({num_features} features)")
        
        if x.shape[2] == output.shape[2]:
            print(f"   ✅ M→M: {num_features} → {num_features} features")
            return True, params
        else:
            print(f"   ❌ M→M FAILED!")
            return False, params
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, 0


def test_forecasting_m2m(model_name, dataset_name, num_features):
    """Test M→M forecasting for PEMS datasets."""
    
    print(f"\n{'─'*50}")
    print(f"Testing: {model_name} on {dataset_name} (Forecasting M→M)")
    print(f"{'─'*50}")
    
    try:
        if model_name == 'MSTN_Transformer':
            from models import MSTN_Transformer as ModelClass
        else:
            from models import MSTN_BiLSTM as ModelClass
        
        # Configuration for forecasting
        class Config:
            task_name = 'long_term_forecast'
            seq_len = 96
            pred_len = 96
            enc_in = num_features
            c_out = num_features
            dropout = 0.1
            num_class = 10
            d_model = 128
            n_heads = 8
            e_layers = 4
            d_ff = 512
        
        config = Config()
        model = ModelClass(config)
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {params:,}")
        
        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, config.seq_len, config.enc_in)
        output = model(x)
        
        print(f"   Input:  {x.shape} ({num_features} sensors)")
        print(f"   Output: {output.shape} ({num_features} sensors)")
        
        if x.shape[2] == output.shape[2]:
            print(f"   ✅ M→M: {num_features} sensors → {num_features} sensors")
            return True, params
        else:
            print(f"   ❌ M→M FAILED!")
            return False, params
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, 0


def main():
    print("\n" + "="*60)
    print("MSTN M→M (Multivariate to Multivariate) TEST")
    print("="*60)
    print("\nVerifying input features = output features for all datasets\n")
    
    # Imputation datasets
    imputation_datasets = [
        ("ETTh1", 7),
        ("ETTh2", 7),
        ("ETTm1", 7),
        ("ETTm2", 7),
        ("ECL", 321),
        ("Weather", 21),
    ]
    
    # Forecasting datasets (PEMS)
    forecasting_datasets = [
        ("PEMS03", 358),
        ("PEMS04", 307),
        ("PEMS07", 883),
        ("PEMS08", 170),
    ]
    
    models = ["MSTN_Transformer", "MSTN_BiLSTM"]
    
    results = []
    
    # Test imputation datasets
    for model in models:
        print(f"\n{'#'*60}")
        print(f"# {model} - IMPUTATION TASKS")
        print(f"{'#'*60}")
        
        for dataset_name, num_features in imputation_datasets:
            success, params = test_imputation_m2m(model, dataset_name, num_features)
            results.append({
                'Model': model,
                'Dataset': dataset_name,
                'Features': num_features,
                'Parameters': params,
                'Success': success,
                'Task': 'Imputation'
            })
    
    # Test forecasting datasets (PEMS)
    for model in models:
        print(f"\n{'#'*60}")
        print(f"# {model} - FORECASTING TASKS (PEMS)")
        print(f"{'#'*60}")
        
        for dataset_name, num_features in forecasting_datasets:
            success, params = test_forecasting_m2m(model, dataset_name, num_features)
            results.append({
                'Model': model,
                'Dataset': dataset_name,
                'Features': num_features,
                'Parameters': params,
                'Success': success,
                'Task': 'Forecasting'
            })
    
    # Summary Table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"\n{'Model':<20} {'Task':<12} {'Dataset':<12} {'Features':<10} {'Parameters':<12} {'M→M':<6}")
    print("-" * 80)
    
    for r in results:
        m2m = "✅" if r['Success'] else "❌"
        print(f"{r['Model']:<20} {r['Task']:<12} {r['Dataset']:<12} {r['Features']:<10} {r['Parameters']:<12,} {m2m:<6}")
    
    # Parameter summary for paper
    print("\n" + "="*60)
    print("PARAMETER SUMMARY (for your paper)")
    print("="*60)
    
    # Find parameters for 7-feature datasets
    trans_7 = None
    bilstm_7 = None
    for r in results:
        if r['Model'] == 'MSTN_Transformer' and r['Features'] == 7 and r['Task'] == 'Imputation':
            trans_7 = r['Parameters']
        if r['Model'] == 'MSTN_BiLSTM' and r['Features'] == 7 and r['Task'] == 'Imputation':
            bilstm_7 = r['Parameters']
    
    if trans_7 and bilstm_7:
        print(f"\n📊 For 7-feature datasets (ETTh1, ETTh2, ETTm1, ETTm2):")
        print(f"   MSTN-Transformer: {trans_7:,} (~{trans_7/1e6:.2f}M)")
        print(f"   MSTN-BiLSTM:      {bilstm_7:,} (~{bilstm_7/1e6:.2f}M)")
    
    # Find PEMS parameters
    print(f"\n📊 For PEMS datasets (Forecasting):")
    for r in results:
        if r['Task'] == 'Forecasting' and r['Model'] == 'MSTN_Transformer':
            print(f"   {r['Dataset']} ({r['Features']} sensors): {r['Parameters']:,} (~{r['Parameters']/1e6:.2f}M)")
    
    print("\n" + "="*60)
    print("✅ ALL M→M TESTS COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
