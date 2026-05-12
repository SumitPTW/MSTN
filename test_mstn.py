#!/usr/bin/env python3
"""
MSTN Imputation M→M Test Script
Shows Multivariate to Multivariate for ETTh1, ECL, Weather
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


def main():
    print("\n" + "="*60)
    print("MSTN IMPUTATION M→M TEST")
    print("="*60)
    print("\nVerifying Multivariate → Multivariate for all datasets\n")
    
    # Datasets and their feature counts
    datasets = [
        ("ETTh1", 7),
        ("ETTh2", 7),
        ("ETTm1", 7),
        ("ETTm2", 7),
        ("ECL", 321),
        ("Weather", 21),
    ]
    
    models = ["MSTN_Transformer", "MSTN_BiLSTM"]
    
    results = []
    
    for model in models:
        print(f"\n{'#'*60}")
        print(f"# {model}")
        print(f"{'#'*60}")
        
        for dataset_name, num_features in datasets:
            success, params = test_imputation_m2m(model, dataset_name, num_features)
            results.append({
                'Model': model,
                'Dataset': dataset_name,
                'Features': num_features,
                'Parameters': params,
                'Success': success
            })
    
    # Summary Table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"\n{'Model':<20} {'Dataset':<12} {'Features':<10} {'Parameters':<12} {'M→M':<6}")
    print("-" * 65)
    
    for r in results:
        m2m = "✅" if r['Success'] else "❌"
        print(f"{r['Model']:<20} {r['Dataset']:<12} {r['Features']:<10} {r['Parameters']:<12,} {m2m:<6}")
    
    # Parameter summary for paper
    print("\n" + "="*60)
    print("PARAMETER SUMMARY (for your paper)")
    print("="*60)
    
    trans_params = None
    bilstm_params = None
    
    for r in results:
        if r['Model'] == 'MSTN_Transformer' and r['Dataset'] == 'ETTh1':
            trans_params = r['Parameters']
        if r['Model'] == 'MSTN_BiLSTM' and r['Dataset'] == 'ETTh1':
            bilstm_params = r['Parameters']
    
    if trans_params and bilstm_params:
        print(f"\n📊 Model Parameter Counts (for 7-feature datasets like ETTh1):")
        print(f"   MSTN-Transformer: {trans_params:,} (~{trans_params/1e6:.2f}M)")
        print(f"   MSTN-BiLSTM:      {bilstm_params:,} (~{bilstm_params/1e6:.2f}M)")
    
    print("\n" + "="*60)
    print("✅ ALL M→M TESTS COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
