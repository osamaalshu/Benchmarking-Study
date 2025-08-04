#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for LSTM-UNet model
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(__file__))

from models.lstmunet import LSTMUNet, create_lstmunet_model

def test_lstmunet():
    """Test the LSTM-UNet model with dummy data"""
    
    print("Testing LSTM-UNet model...")
    
    # Create model
    model = create_lstmunet_model(
        image_size=(256, 256),
        in_channels=3,
        out_channels=3,
        base_filters=32,  # Smaller for testing
        depth=3,          # Smaller for testing
        lstm_hidden_channels=32,
        lstm_layers=1,
        dropout_rate=0.1
    )
    
    print(f"Model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with dummy data
    batch_size = 8  # Changed from 2 to 8
    channels = 3
    height = 256
    width = 256
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, channels, height, width)
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, 3, {height}, {width})")
    
    # Check if output shape is correct
    expected_shape = (batch_size, 3, height, width)
    if output.shape == expected_shape:
        print("✅ LSTM-UNet test passed!")
        return True
    else:
        print(f"❌ LSTM-UNet test failed! Expected {expected_shape}, got {output.shape}")
        return False

def test_lstmunet_sequence():
    """Test the LSTM-UNet model with sequence data"""
    
    print("\nTesting LSTM-UNet model with sequence data...")
    
    # Create model
    model = create_lstmunet_model(
        image_size=(128, 128),
        in_channels=3,
        out_channels=3,
        base_filters=16,  # Smaller for testing
        depth=2,          # Smaller for testing
        lstm_hidden_channels=16,
        lstm_layers=1,
        dropout_rate=0.1
    )
    
    # Test with sequence data
    batch_size = 8  # Changed from 1 to 8
    seq_len = 3
    channels = 3
    height = 128
    width = 128
    
    # Create dummy sequence input
    dummy_sequence = torch.randn(batch_size, seq_len, channels, height, width)
    print(f"Sequence input shape: {dummy_sequence.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_sequence)
    
    print(f"Sequence output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, 3, {height}, {width})")
    
    # Check if output shape is correct
    expected_shape = (batch_size, 3, height, width)
    if output.shape == expected_shape:
        print("✅ LSTM-UNet sequence test passed!")
        return True
    else:
        print(f"❌ LSTM-UNet sequence test failed! Expected {expected_shape}, got {output.shape}")
        return False

if __name__ == "__main__":
    print("Running LSTM-UNet tests...")
    
    test1_passed = test_lstmunet()
    test2_passed = test_lstmunet_sequence()
    
    if test1_passed and test2_passed:
        print("\n🎉 All LSTM-UNet tests passed!")
    else:
        print("\n❌ Some LSTM-UNet tests failed!")
        sys.exit(1) 