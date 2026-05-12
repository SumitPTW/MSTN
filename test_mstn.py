#!/usr/bin/env python3
"""
MSTN M→M (Multivariate to Multivariate) Test Script
Based on YOUR paper's actual datasets (PEMS forecasting)
"""
import sys
import torch

sys.path.append('.')

def test_mstn_transformer_pems_m2m():
    """Test MSTN_Transformer for M→M on PEMS03 (358 sensors)."""
    print("="*60)
    print("Testing MSTN_Transformer (PEMS03 M→M)")
    print("="*60)
    
    try:
        from models import MSTN_Transformer
        
        # Use PEMS03 configuration (YOUR actual dataset)
        class MockConfig:
            task_name = 'long_term_forecast'  # Forecasting task
            seq_len = 96
            pred_len = 96
            enc_in = 358   # PEMS03: 358 sensors
            c_out = 358    # Output: ALL 358 sensors (M→M)
            dropout = 0.1
            num_class = 10
            d_model = 128
            n_heads = 8
            e_layers = 4
            d_ff = 512
        
        config = MockConfig()
        model = MSTN_Transformer(config)
        print(f"✅ MSTN_Transformer created successfully")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
        
        batch_size = 4
        x = torch.randn(batch_size, config.seq_len, config.enc_in)
        output = model(x)
        
        print(f"\n   Input shape:  {x.shape}")
        print(f"   → {x.shape[2]} sensors (ALL PEMS03 sensors)")
        print(f"\n   Output shape: {output.shape}")
        print(f"   → {output.shape[2]} sensors (ALL PEMS03 sensors)")
        
        # M→M Verification
        print(f"\n{'─'*40}")
        print(f"📊 M→M VERIFICATION:")
        print(f"   Input sensors:  {x.shape[2]}")
        print(f"   Output sensors: {output.shape[2]}")
        
        if x.shape[2] == output.shape[2]:
            print(f"   ✅ M→M: {x.shape[2]} sensors → {output.shape[2]} sensors")
            print(f"\n✅ PEMS03 M→M Test PASSED!")
            return True
        else:
            print(f"   ❌ M→M FAILED: Feature mismatch!")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_mstn_transformer_pems04_m2m():
    """Test MSTN_Transformer for M→M on PEMS04 (307 sensors)."""
    print("\n" + "="*60)
    print("Testing MSTN_Transformer (PEMS04 M→M)")
    print("="*60)
    
    try:
        from models import MSTN_Transformer
        
        class MockConfig:
            task_name = 'long_term_forecast'
            seq_len = 96
            pred_len = 96
            enc_in = 307   # PEMS04: 307 sensors
            c_out = 307    # Output: ALL 307 sensors
            dropout = 0.1
            num_class = 10
            d_model = 128
            n_heads = 8
            e_layers = 4
            d_ff = 512
        
        config = MockConfig()
        model = MSTN_Transformer(config)
        
        batch_size = 4
        x = torch.randn(batch_size, config.seq_len, config.enc_in)
        output = model(x)
        
        print(f"   Input:  {x.shape} (307 sensors)")
        print(f"   Output: {output.shape} (307 sensors)")
        
        if x.shape[2] == output.shape[2]:
            print(f"   ✅ M→M: 307 sensors → 307 sensors")
            return True
        else:
            print(f"   ❌ M→M FAILED!")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_mstn_bilstm_pems_m2m():
    """Test MSTN_BiLSTM for M→M on PEMS03."""
    print("\n" + "="*60)
    print("Testing MSTN_BiLSTM (PEMS03 M→M)")
    print("="*60)
    
    try:
        from models import MSTN_BiLSTM
        
        class MockConfig:
            task_name = 'long_term_forecast'
            seq_len = 96
            pred_len = 96
            enc_in = 358
            c_out = 358
            dropout = 0.1
            num_class = 10
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
        
        print(f"\n   Input:  {x.shape} (358 sensors)")
        print(f"   Output: {output.shape} (358 sensors)")
        
        if x.shape[2] == output.shape[2]:
            print(f"   ✅ M→M: 358 sensors → 358 sensors")
            return True
        else:
            print(f"   ❌ M→M FAILED!")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_weather_imputation_m2m():
    """Test M→M on Weather imputation (21 features)."""
    print("\n" + "="*60)
    print("Testing Weather Imputation M→M (21 features)")
    print("="*60)
    
    try:
        from models import MSTN_Transformer
        
        class MockConfig:
            task_name = 'imputation'  # Imputation task
            seq_len = 96
            pred_len = 96
            enc_in = 21    # Weather: 21 features
            c_out = 21     # Output: ALL 21 features
            dropout = 0.1
            num_class = 10
            d_model = 128
            n_heads = 8
            e_layers = 4
            d_ff = 512
        
        config = MockConfig()
        model = MSTN_Transformer(config)
        
        batch_size = 4
        x = torch.randn(batch_size, config.seq_len, config.enc_in)
        output = model(x)
        
        print(f"   Input:  {x.shape} (21 weather variables)")
        print(f"   Output: {output.shape} (21 weather variables)")
        
        if x.shape[2] == output.shape[2]:
            print(f"   ✅ M→M: 21 features → 21 features")
            return True
        else:
            print(f"   ❌ M→M FAILED!")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Run all M→M tests based on YOUR paper."""
    print("\n" + "="*60)
    print("MSTN M→M VERIFICATION TESTS")
    print("Based on YOUR paper's datasets")
    print("="*60)
    
    print("\n📊 Testing PEMS Forecasting (M→M):")
    success_pems03_trans = test_mstn_transformer_pems_m2m()
    success_pems04_trans = test_mstn_transformer_pems04_m2m()
    success_pems03_bilstm = test_mstn_bilstm_pems_m2m()
    
    print("\n📊 Testing Weather Imputation (M→M):")
    success_weather = test_weather_imputation_m2m()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_success = success_pems03_trans and success_pems04_trans and success_pems03_bilstm and success_weather
    
    if all_success:
        print("✅ ALL TESTS PASSED!")
        print("\n📊 M→M Verification Successful for YOUR paper:")
        print("   ✓ PEMS03 Forecasting:   358 sensors → 358 sensors")
        print("   ✓ PEMS04 Forecasting:   307 sensors → 307 sensors")
        print("   ✓ PEMS03 BiLSTM:        358 sensors → 358 sensors")
        print("   ✓ Weather Imputation:   21 features → 21 features")
        print("\n✅ MSTN correctly implements Multivariate to Multivariate (M→M)")
    else:
        print("❌ Some tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
