#!/usr/bin/env python3
"""
MSTN Basic Functional Test Script.
"""
import sys
import torch

sys.path.append('.')

def test_mstn_transformer():
    """Test MSTN_Transformer model creation and forward pass."""
    print("="*60)
    print("Testing MSTN_Transformer")
    print("="*60)
    
    try:
        from models import MSTN_Transformer
        
        class MockConfig:
            task_name = 'classification'
            seq_len = 96
            enc_in = 7
            num_class = 10
            dropout = 0.1
            pred_len = 96
            c_out = 7
            d_model = 128
            n_heads = 8
            e_layers = 4
            d_ff = 512
        
        config = MockConfig()
        
        # Create model
        model = MSTN_Transformer(config)
        print(f"✅ MSTN_Transformer created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
        
        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, config.seq_len, config.enc_in)
        output = model(x)
        
        print(f"✅ Forward pass successful")
        print(f"   Input shape:  {x.shape}")
        print(f"   Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ MSTN_Transformer test failed: {e}")
        return False


def test_mstn_bilstm():
    """Test MSTN_BiLSTM model creation and forward pass."""
    print("\n" + "="*60)
    print("Testing MSTN_BiLSTM")
    print("="*60)
    
    try:
        from models import MSTN_BiLSTM
        
        class MockConfig:
            task_name = 'classification'
            seq_len = 96
            enc_in = 7
            num_class = 10
            dropout = 0.1
            pred_len = 96
            c_out = 7
            d_model = 128
            n_heads = 8
            e_layers = 4
            d_ff = 512
        
        config = MockConfig()
        
        model = MSTN_BiLSTM(config)
        print(f"✅ MSTN_BiLSTM created successfully")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
        
        batch_size = 4
        x = torch.randn(batch_size, config.seq_len, config.enc_in)
        output = model(x)
        
        print(f"✅ Forward pass successful")
        print(f"   Input shape:  {x.shape}")
        print(f"   Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ MSTN_BiLSTM test failed: {e}")
        return False


# ============================================================
#
# ============================================================

def test_mstn_transformer_forecasting():
    """Test MSTN_Transformer for forecasting task."""
    print("\n" + "="*60)
    print("Testing MSTN_Transformer (Forecasting)")
    print("="*60)
    
    try:
        from models import MSTN_Transformer
        
        class MockConfig:
            task_name = 'long_term_forecast'
            seq_len = 96
            pred_len = 96
            enc_in = 7
            c_out = 7
            dropout = 0.1
        
        config = MockConfig()
        model = MSTN_Transformer(config)
        x = torch.randn(4, config.seq_len, config.enc_in)
        output = model(x)
        
        print(f"✅ Forecasting output shape: {output.shape}")
        assert output.shape == (4, config.pred_len, config.c_out)
        print(f"   Input shape:  {x.shape}")
        print(f"   Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ MSTN_Transformer forecasting test failed: {e}")
        return False


def test_mstn_transformer_imputation():
    """Test MSTN_Transformer for imputation task."""
    print("\n" + "="*60)
    print("Testing MSTN_Transformer (Imputation)")
    print("="*60)
    
    try:
        from models import MSTN_Transformer
        
        class MockConfig:
            task_name = 'imputation'
            seq_len = 96
            enc_in = 7
            c_out = 7
            dropout = 0.1
        
        config = MockConfig()
        model = MSTN_Transformer(config)
        x = torch.randn(4, config.seq_len, config.enc_in)
        output = model(x)
        
        print(f"✅ Imputation output shape: {output.shape}")
        assert output.shape == (4, config.seq_len, config.c_out)
        print(f"   Input shape:  {x.shape}")
        print(f"   Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ MSTN_Transformer imputation test failed: {e}")
        return False


def test_mstn_bilstm_forecasting():
    """Test MSTN_BiLSTM for forecasting task."""
    print("\n" + "="*60)
    print("Testing MSTN_BiLSTM (Forecasting)")
    print("="*60)
    
    try:
        from models import MSTN_BiLSTM
        
        class MockConfig:
            task_name = 'long_term_forecast'
            seq_len = 96
            pred_len = 96
            enc_in = 7
            c_out = 7
            dropout = 0.1
        
        config = MockConfig()
        model = MSTN_BiLSTM(config)
        x = torch.randn(4, config.seq_len, config.enc_in)
        output = model(x)
        
        print(f"✅ BiLSTM Forecasting output shape: {output.shape}")
        assert output.shape == (4, config.pred_len, config.c_out)
        print(f"   Input shape:  {x.shape}")
        print(f"   Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ MSTN_BiLSTM forecasting test failed: {e}")
        return False


def test_mstn_bilstm_imputation():
    """Test MSTN_BiLSTM for imputation task."""
    print("\n" + "="*60)
    print("Testing MSTN_BiLSTM (Imputation)")
    print("="*60)
    
    try:
        from models import MSTN_BiLSTM
        
        class MockConfig:
            task_name = 'imputation'
            seq_len = 96
            enc_in = 7
            c_out = 7
            dropout = 0.1
        
        config = MockConfig()
        model = MSTN_BiLSTM(config)
        x = torch.randn(4, config.seq_len, config.enc_in)
        output = model(x)
        
        print(f"✅ BiLSTM Imputation output shape: {output.shape}")
        assert output.shape == (4, config.seq_len, config.c_out)
        print(f"   Input shape:  {x.shape}")
        print(f"   Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ MSTN_BiLSTM imputation test failed: {e}")
        return False


# ============================================================
# MAIN FUNCTION 
# ============================================================

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("MSTN Basic Functional Tests")
    print("="*60)
    
    # Test both model variants (classification)
    success_transformer = test_mstn_transformer()
    success_bilstm = test_mstn_bilstm()
    
    # Test additional tasks
    success_trans_forecast = test_mstn_transformer_forecasting()
    success_trans_impute = test_mstn_transformer_imputation()
    success_bilstm_forecast = test_mstn_bilstm_forecasting()
    success_bilstm_impute = test_mstn_bilstm_imputation()
    
    # Final summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_success = (success_transformer and success_bilstm and 
                   success_trans_forecast and success_trans_impute and
                   success_bilstm_forecast and success_bilstm_impute)
    
    if all_success:
        print("✅ All tests passed! MSTN models work for:")
        print("   - Classification (Transformer & BiLSTM)")
        print("   - Forecasting (Transformer & BiLSTM)")
        print("   - Imputation (Transformer & BiLSTM)")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
