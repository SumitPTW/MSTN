#!/usr/bin/env python3
"""
MSTN Basic Functional Test Script.
"""
import sys
import torch

# Ensure the local modules can be imported
sys.path.append('.')

def test_mstn_transformer():
    """Test MSTN_Transformer model creation and forward pass."""
    print("="*60)
    print("Testing MSTN_Transformer")
    print("="*60)
    
    try:
        # Import your model
        from models import MSTN_Transformer
        
        # Create a mock configuration object
        class MockConfig:
            task_name = 'classification'
            seq_len = 96
            enc_in = 7
            num_class = 10
            dropout = 0.1
            # Add other attributes your model __init__ expects
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

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("MSTN Basic Functional Tests")
    print("="*60)
    
    # Test both model variants
    success_transformer = test_mstn_transformer()
    success_bilstm = test_mstn_bilstm()
    
    # Final summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if success_transformer and success_bilstm:
        print("✅ Tests passed! MSTN")
        print("forward passes.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
