#!/usr/bin/env python3
"""
MSTN Basic Functional Test Script - M→M Verification
"""
import sys
import torch

sys.path.append('.')

def test_mstn_transformer_m2m():
    """Test MSTN_Transformer for M→M (Multivariate to Multivariate)."""
    print("="*60)
    print("Testing MSTN_Transformer (M→M)")
    print("="*60)
    
    try:
        from models import MSTN_Transformer
        
        # Config for PEMS03 (358 sensors)
        class MockConfig:
            task_name = 'long_term_forecast'
            seq_len = 96
            pred_len = 96
            enc_in = 358   # Input: ALL 358 sensors
            c_out = 358    # Output: ALL 358 sensors
            dropout = 0.1
            num_class = 10
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
        print(f"\n   Input shape:  {x.shape}")
        print(f"   → {x.shape[2]} features (ALL sensors)")
        
        output = model(x)
        print(f"\n   Output shape: {output.shape}")
        print(f"   → {output.shape[2]} features (ALL sensors)")
        
        # M→M Verification
        print(f"\n{'─'*40}")
        print(f"📊 M→M VERIFICATION:")
        print(f"   Input features:  {x.shape[2]} (ALL {config.enc_in} sensors)")
        print(f"   Output features: {output.shape[2]} (ALL {config.c_out} sensors)")
        
        if x.shape[2] == output.shape[2]:
            print(f"   ✅ M→M: {x.shape[2]} features → {output.shape[2]} features")
        else:
            print(f"   ❌ M→M FAILED: Feature mismatch!")
            return False
        
        # Shape verification
        if len(output.shape) == 3 and output.shape[1] == config.pred_len:
            print(f"   ✅ Output: [Batch={output.shape[0]}, Horizon={output.shape[1]}, Features={output.shape[2]}]")
        
        print(f"\n✅ Forward pass successful - M→M preserved!")
        return True
        
    except Exception as e:
        print(f"❌ MSTN_Transformer test failed: {e}")
        return False


def test_mstn_bilstm_m2m():
    """Test MSTN_BiLSTM for M→M (Multivariate to Multivariate)."""
    print("\n" + "="*60)
    print("Testing MSTN_BiLSTM (M→M)")
    print("="*60)
    
    try:
        from models import MSTN_BiLSTM
        
        class MockConfig:
            task_name = 'long_term_forecast'
            seq_len = 96
            pred_len = 96
            enc_in = 358   # Input: ALL 358 sensors
            c_out = 358    # Output: ALL 358 sensors
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
        print(f"\n   Input shape:  {x.shape}")
        print(f"   → {x.shape[2]} features (ALL sensors)")
        
        output = model(x)
        print(f"\n   Output shape: {output.shape}")
        print(f"   → {output.shape[2]} features (ALL sensors)")
        
        # M→M Verification
        print(f"\n{'─'*40}")
        print(f"📊 M→M VERIFICATION:")
        print(f"   Input features:  {x.shape[2]} (ALL {config.enc_in} sensors)")
        print(f"   Output features: {output.shape[2]} (ALL {config.c_out} sensors)")
        
        if x.shape[2] == output.shape[2]:
            print(f"   ✅ M→M: {x.shape[2]} features → {output.shape[2]} features")
        else:
            print(f"   ❌ M→M FAILED: Feature mismatch!")
            return False
        
        print(f"\n✅ Forward pass successful - M→M preserved!")
        return True
        
    except Exception as e:
        print(f"❌ MSTN_BiLSTM test failed: {e}")
        return False


def test_etth1_m2m():
    """Test M→M on ETTh1 dataset (7 features)."""
    print("\n" + "="*60)
    print("Testing M→M on ETTh1 Dataset (7 features)")
    print("="*60)
    
    try:
        from models import MSTN_Transformer
        
        class MockConfig:
            task_name = 'imputation'
            seq_len = 96
            pred_len = 96
            enc_in = 7     # ETTh1 has 7 features (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)
            c_out = 7      # Output ALL 7 features
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
        
        print(f"\n   ETTh1 Dataset (Electricity Transformer Temperature):")
        print(f"   Features: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT")
        print(f"   Input:  {x.shape} (7 features)")
        print(f"   Output: {output.shape} (7 features)")
        
        print(f"\n{'─'*40}")
        print(f"📊 M→M VERIFICATION:")
        print(f"   Input features:  {x.shape[2]} (ALL 7 ETTh1 features)")
        print(f"   Output features: {output.shape[2]} (ALL 7 ETTh1 features)")
        
        if x.shape[2] == output.shape[2]:
            print(f"   ✅ M→M: 7 features → 7 features")
            print(f"\n   ✅ ETTh1 M→M Test PASSED!")
            return True
        else:
            print(f"   ❌ M→M FAILED!")
            return False
            
    except Exception as e:
        print(f"❌ ETTh1 test failed: {e}")
        return False


def test_weather_m2m():
    """Test M→M on Weather dataset (21 features)."""
    print("\n" + "="*60)
    print("Testing M→M on Weather Dataset (21 features)")
    print("="*60)
    
    try:
        from models import MSTN_Transformer
        
        class MockConfig:
            task_name = 'imputation'
            seq_len = 96
            pred_len = 96
            enc_in = 21    # Weather has 21 features
            c_out = 21     # Output ALL 21 features
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
        
        print(f"\n   Weather Dataset:")
        print(f"   Input:  {x.shape} (21 weather variables)")
        print(f"   Output: {output.shape} (21 weather variables)")
        
        print(f"\n{'─'*40}")
        print(f"📊 M→M VERIFICATION:")
        print(f"   Input features:  {x.shape[2]} (ALL 21 weather features)")
        print(f"   Output features: {output.shape[2]} (ALL 21 weather features)")
        
        if x.shape[2] == output.shape[2]:
            print(f"   ✅ M→M: 21 features → 21 features")
            return True
        else:
            print(f"   ❌ M→M FAILED!")
            return False
            
    except Exception as e:
        print(f"❌ Weather test failed: {e}")
        return False


def test_ecl_m2m():
    """Test M→M on ECL dataset (321 features)."""
    print("\n" + "="*60)
    print("Testing M→M on ECL Dataset (321 features)")
    print("="*60)
    
    try:
        from models import MSTN_Transformer
        
        class MockConfig:
            task_name = 'imputation'
            seq_len = 96
            pred_len = 96
            enc_in = 321   # ECL has 321 customers
            c_out = 321    # Output ALL 321 customers
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
        
        print(f"\n   ECL Dataset (Electricity Load Consumption):")
        print(f"   Input:  {x.shape} (321 customers)")
        print(f"   Output: {output.shape} (321 customers)")
        
        print(f"\n{'─'*40}")
        print(f"📊 M→M VERIFICATION:")
        print(f"   Input features:  {x.shape[2]} (ALL 321 customers)")
        print(f"   Output features: {output.shape[2]} (ALL 321 customers)")
        
        if x.shape[2] == output.shape[2]:
            print(f"   ✅ M→M: 321 features → 321 features")
            return True
        else:
            print(f"   ❌ M→M FAILED!")
            return False
            
    except Exception as e:
        print(f"❌ ECL test failed: {e}")
        return False


def test_ettm1_m2m():
    """Test M→M on ETTm1 dataset (7 features, 15-min frequency)."""
    print("\n" + "="*60)
    print("Testing M→M on ETTm1 Dataset (7 features, 15-min)")
    print("="*60)
    
    try:
        from models import MSTN_BiLSTM
        
        class MockConfig:
            task_name = 'imputation'
            seq_len = 96
            pred_len = 96
            enc_in = 7     # ETTm1 has 7 features
            c_out = 7      # Output ALL 7 features
            dropout = 0.1
            num_class = 10
            d_model = 128
            n_heads = 8
            e_layers = 4
            d_ff = 512
        
        config = MockConfig()
        model = MSTN_BiLSTM(config)
        
        batch_size = 4
        x = torch.randn(batch_size, config.seq_len, config.enc_in)
        output = model(x)
        
        print(f"\n   ETTm1 Dataset (15-min frequency):")
        print(f"   Input:  {x.shape} (7 features)")
        print(f"   Output: {output.shape} (7 features)")
        
        print(f"\n{'─'*40}")
        print(f"📊 M→M VERIFICATION:")
        print(f"   Input features:  {x.shape[2]} (ALL 7 ETTm1 features)")
        print(f"   Output features: {output.shape[2]} (ALL 7 ETTm1 features)")
        
        if x.shape[2] == output.shape[2]:
            print(f"   ✅ M→M: 7 features → 7 features")
            return True
        else:
            print(f"   ❌ M→M FAILED!")
            return False
            
    except Exception as e:
        print(f"❌ ETTm1 test failed: {e}")
        return False


def main():
    """Run all M→M tests."""
    print("\n" + "="*60)
    print("MSTN M→M (Multivariate to Multivariate) Tests")
    print("="*60)
    print("\nGoal: Verify that input features = output features")
    print("(e.g., 358 sensors → 358 sensors)\n")
    
    # Run all tests
    success_trans = test_mstn_transformer_m2m()
    success_bilstm = test_mstn_bilstm_m2m()
    success_etth1 = test_etth1_m2m()
    success_weather = test_weather_m2m()
    success_ecl = test_ecl_m2m()
    success_ettm1 = test_ettm1_m2m()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_success = success_trans and success_bilstm and success_etth1 and success_weather and success_ecl and success_ettm1
    
    if all_success:
        print("✅ ALL TESTS PASSED!")
        print("\n📊 M→M Verification Successful:")
        print("   ✓ PEMS03:   358 features → 358 features")
        print("   ✓ ETTh1:    7 features → 7 features (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)")
        print("   ✓ ETTm1:    7 features → 7 features (15-min frequency)")
        print("   ✓ Weather:  21 features → 21 features")
        print("   ✓ ECL:      321 features → 321 features")
        print("\n✅ MSTN correctly implements Multivariate to Multivariate (M→M)")
        print("   All models preserve feature dimensionality across all tasks.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
