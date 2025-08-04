#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM-UNet for 2D Image Segmentation
Based on: https://gitlab.com/shaked0/lstmUnet

This implementation adapts the ConvLSTM concept to 2D images using PyTorch,
combining the benefits of LSTM temporal modeling with U-Net spatial modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


class ConvLSTM2DCell(nn.Module):
    """
    2D Convolutional LSTM Cell
    
    This cell implements the core LSTM mechanism using 2D convolutions
    instead of fully connected layers, making it suitable for spatial data.
    """
    
    def __init__(self, 
                 input_channels: int,
                 hidden_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 bias: bool = True,
                 activation: str = 'tanh',
                 recurrent_activation: str = 'sigmoid'):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Activation functions
        self.activation = getattr(torch, activation) if hasattr(torch, activation) else torch.tanh
        self.recurrent_activation = getattr(torch, recurrent_activation) if hasattr(torch, recurrent_activation) else torch.sigmoid
        
        # Input convolution (for input x_t)
        self.input_conv = nn.Conv2d(
            input_channels, 
            hidden_channels * 4,  # 4 gates: input, forget, cell, output
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        
        # Recurrent convolution (for hidden state h_{t-1})
        self.recurrent_conv = nn.Conv2d(
            hidden_channels,
            hidden_channels * 4,  # 4 gates: input, forget, cell, output
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False  # No bias for recurrent connections
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Forward pass of the ConvLSTM cell
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            hidden_state: Tuple of (h_prev, c_prev) or None for initial state
            
        Returns:
            Tuple of (h_new, c_new)
        """
        batch_size, _, height, width = x.size()
        
        # Initialize hidden state if not provided
        if hidden_state is None or hidden_state[0] is None or hidden_state[1] is None:
            # Ensure device compatibility for Colab (CUDA/MPS/CPU)
            device = x.device
            dtype = x.dtype
            h_prev = torch.zeros(batch_size, self.hidden_channels, height, width, device=device, dtype=dtype)
            c_prev = torch.zeros(batch_size, self.hidden_channels, height, width, device=device, dtype=dtype)
        else:
            h_prev, c_prev = hidden_state
        
        # Compute gates
        x_conv = self.input_conv(x)
        h_conv = self.recurrent_conv(h_prev)
        
        # Split the convolutions into individual gates
        x_i, x_f, x_c, x_o = torch.split(x_conv, self.hidden_channels, dim=1)
        h_i, h_f, h_c, h_o = torch.split(h_conv, self.hidden_channels, dim=1)
        
        # LSTM equations
        i_t = self.recurrent_activation(x_i + h_i)  # Input gate
        f_t = self.recurrent_activation(x_f + h_f)  # Forget gate
        c_tilde = self.activation(x_c + h_c)        # Candidate cell state
        c_new = f_t * c_prev + i_t * c_tilde       # New cell state
        o_t = self.recurrent_activation(x_o + h_o)  # Output gate
        h_new = o_t * self.activation(c_new)        # New hidden state
        
        return h_new, c_new


class ConvLSTM2D(nn.Module):
    """
    2D Convolutional LSTM Layer
    
    This layer applies ConvLSTM cells across a sequence of 2D images.
    """
    
    def __init__(self, 
                 input_channels: int,
                 hidden_channels: int,
                 kernel_size: int = 3,
                 num_layers: int = 1,
                 batch_first: bool = True,
                 return_sequences: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.dropout = dropout
        
        # Create ConvLSTM cells
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                cell = ConvLSTM2DCell(input_channels, hidden_channels, kernel_size)
            else:
                cell = ConvLSTM2DCell(hidden_channels, hidden_channels, kernel_size)
            self.cells.append(cell)
        
        # Dropout layer
        if dropout > 0:
            self.dropout_layer = nn.Dropout2d(dropout)
        else:
            self.dropout_layer = None
    
    def forward(self, x: torch.Tensor, hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None):
        """
        Forward pass of the ConvLSTM layer
        
        Args:
            x: Input tensor of shape (batch, seq_len, channels, height, width) if batch_first=True
               or (seq_len, batch, channels, height, width) if batch_first=False
            hidden_state: List of (h, c) tuples for each layer or None
            
        Returns:
            Output tensor and final hidden states
        """
        if not self.batch_first:
            x = x.transpose(0, 1)  # Make batch first
        
        batch_size, seq_len, channels, height, width = x.size()
        
        # Initialize hidden states if not provided
        if hidden_state is None:
            hidden_state = [None] * self.num_layers
        
        # Process sequence through layers
        layer_outputs = []
        current_input = x
        
        for layer_idx in range(self.num_layers):
            layer_output = []
            if hidden_state[layer_idx] is None:
                h, c = None, None
            else:
                h, c = hidden_state[layer_idx]
            
            for t in range(seq_len):
                h, c = self.cells[layer_idx](current_input[:, t], (h, c))
                layer_output.append(h)
            
            # Stack outputs for this layer
            layer_output = torch.stack(layer_output, dim=1)  # (batch, seq_len, hidden_channels, height, width)
            
            # Apply dropout if specified
            if self.dropout_layer is not None and layer_idx < self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)
            
            layer_outputs.append(layer_output)
            current_input = layer_output
            hidden_state[layer_idx] = (h, c)
        
        # Return final layer output
        output = layer_outputs[-1]
        
        if not self.return_sequences:
            output = output[:, -1]  # Take only the last timestep
        
        return output, hidden_state


class LSTMUNet(nn.Module):
    """
    LSTM-UNet for 2D Image Segmentation
    
    This model combines the temporal modeling capabilities of LSTM with
    the spatial modeling capabilities of U-Net for sequence-based segmentation.
    """
    
    def __init__(self, 
                 in_channels: int = 3,
                 out_channels: int = 3,  # background, interior, boundary
                 base_filters: int = 64,
                 depth: int = 4,
                 lstm_hidden_channels: int = 64,
                 lstm_layers: int = 2,
                 lstm_kernel_size: int = 3,
                 dropout_rate: float = 0.1,
                 use_batch_norm: bool = True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.depth = depth
        self.lstm_hidden_channels = lstm_hidden_channels
        self.lstm_layers = lstm_layers
        self.lstm_kernel_size = lstm_kernel_size
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Calculate filter sizes for each depth level
        self.filters = [base_filters]
        for i in range(depth - 1):
            self.filters.append(self.filters[-1] * 2)
        
        print(f"LSTM-UNet filters: {self.filters}")
        
        # Print device information
        device = get_device()
        print(f"LSTM-UNet will use device: {device}")
        if device.type == "cuda":
            print(f"✅ CUDA GPU: {torch.cuda.get_device_name()}")
        elif device.type == "mps":
            print("✅ Apple Silicon GPU (MPS)")
        else:
            print("⚠️  CPU (no GPU acceleration)")
        
        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        self.encoder_lstm = nn.ModuleList()
        
        # Initial convolution
        self.initial_conv = self._make_conv_block(
            in_channels, base_filters, use_batch_norm, dropout_rate
        )
        
        # Encoder blocks
        for i in range(depth - 1):
            # Downsampling
            down_conv = self._make_conv_block(
                self.filters[i], self.filters[i + 1], use_batch_norm, dropout_rate
            )
            self.encoder.append(down_conv)
            
            # LSTM layer for temporal modeling
            lstm_layer = ConvLSTM2D(
                input_channels=self.filters[i + 1],
                hidden_channels=self.filters[i + 1],  # Match the number of channels
                kernel_size=lstm_kernel_size,
                num_layers=lstm_layers,
                dropout=dropout_rate
            )
            self.encoder_lstm.append(lstm_layer)
        
        # Bottleneck
        self.bottleneck = self._make_conv_block(
            self.filters[-1], self.filters[-1] * 2, use_batch_norm, dropout_rate
        )
        self.bottleneck_lstm = ConvLSTM2D(
            input_channels=self.filters[-1] * 2,
            hidden_channels=self.filters[-1] * 2,  # Match the number of channels
            kernel_size=lstm_kernel_size,
            num_layers=lstm_layers,
            dropout=dropout_rate
        )
        
        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        self.decoder_lstm = nn.ModuleList()
        
        for i in range(depth - 1, 0, -1):
            # Upsampling
            up_conv = self._make_up_block(
                self.filters[i] * 2, self.filters[i - 1], use_batch_norm, dropout_rate
            )
            self.decoder.append(up_conv)
            
            # LSTM layer for temporal modeling (after concatenation, channels will be 2 * filters[i-1])
            lstm_layer = ConvLSTM2D(
                input_channels=self.filters[i - 1] * 2,  # After concatenation
                hidden_channels=self.filters[i - 1] * 2,  # Match the number of channels
                kernel_size=lstm_kernel_size,
                num_layers=lstm_layers,
                dropout=dropout_rate
            )
            self.decoder_lstm.append(lstm_layer)
        
        # Final output layer (after last concatenation, channels will be 2 * base_filters)
        self.final_conv = nn.Conv2d(base_filters * 2, out_channels, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_conv_block(self, in_channels: int, out_channels: int, use_batch_norm: bool, dropout_rate: float):
        """Create a convolutional block with optional batch norm and dropout"""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        
        if use_batch_norm:
            layers.insert(1, nn.BatchNorm2d(out_channels))
        
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        
        return nn.Sequential(*layers)
    
    def _make_up_block(self, in_channels: int, out_channels: int, use_batch_norm: bool, dropout_rate: float):
        """Create an upsampling block"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass of the LSTM-UNet
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
               For sequence processing, this should be the current frame
               
        Returns:
            Output tensor of shape (batch, out_channels, height, width)
        """
        # Add sequence dimension if not present (for single frame processing)
        if len(x.shape) == 4:
            x = x.unsqueeze(1)  # (batch, 1, channels, height, width)
        
        batch_size, seq_len, channels, height, width = x.shape
        
        # Process each timestep
        encoder_outputs = []
        current_input = x
        
        # Initial convolution
        initial_output = []
        for t in range(seq_len):
            out = self.initial_conv(current_input[:, t])
            initial_output.append(out)
        current_input = torch.stack(initial_output, dim=1)
        encoder_outputs.append(current_input)
        
        # Encoder path with LSTM
        for i, (encoder_block, lstm_layer) in enumerate(zip(self.encoder, self.encoder_lstm)):
            # Convolutional processing
            conv_output = []
            for t in range(seq_len):
                out = encoder_block(current_input[:, t])
                out = F.max_pool2d(out, 2)  # Downsample
                conv_output.append(out)
            current_input = torch.stack(conv_output, dim=1)
            
            # LSTM processing
            lstm_output, _ = lstm_layer(current_input)
            current_input = lstm_output
            
            encoder_outputs.append(current_input)
        
        # Bottleneck
        bottleneck_output = []
        for t in range(seq_len):
            out = self.bottleneck(current_input[:, t])
            bottleneck_output.append(out)
        current_input = torch.stack(bottleneck_output, dim=1)
        
        # LSTM at bottleneck
        lstm_output, _ = self.bottleneck_lstm(current_input)
        current_input = lstm_output
        
        # Decoder path with LSTM
        for i, (decoder_block, lstm_layer) in enumerate(zip(self.decoder, self.decoder_lstm)):
            # Skip connection
            skip_connection = encoder_outputs[-(i + 2)]
            
            # Upsample and concatenate
            up_output = []
            for t in range(seq_len):
                out = decoder_block(current_input[:, t])
                # Resize skip connection to match current size
                skip = F.interpolate(skip_connection[:, t], size=out.shape[2:], mode='bilinear', align_corners=False)
                out = torch.cat([out, skip], dim=1)
                up_output.append(out)
            current_input = torch.stack(up_output, dim=1)
            
            # LSTM processing
            lstm_output, _ = lstm_layer(current_input)
            current_input = lstm_output
        
        # Final convolution (use last timestep for single frame output)
        final_output = self.final_conv(current_input[:, -1])
        
        return final_output


def get_device():
    """Get the best available device (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def create_lstmunet_model(image_size: Tuple[int, int], 
                         in_channels: int = 3,
                         out_channels: int = 3,
                         base_filters: int = 64,
                         depth: int = 4,
                         lstm_hidden_channels: int = 64,
                         lstm_layers: int = 2,
                         dropout_rate: float = 0.1) -> LSTMUNet:
    """
    Create an LSTM-UNet model with specified parameters
    
    Args:
        image_size: Tuple of (height, width) for input images
        in_channels: Number of input channels
        out_channels: Number of output channels (classes)
        base_filters: Number of base filters
        depth: Depth of the U-Net
        lstm_hidden_channels: Number of hidden channels in LSTM layers
        lstm_layers: Number of LSTM layers
        dropout_rate: Dropout rate
        
    Returns:
        Configured LSTM-UNet model
    """
    model = LSTMUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_filters=base_filters,
        depth=depth,
        lstm_hidden_channels=lstm_hidden_channels,
        lstm_layers=lstm_layers,
        dropout_rate=dropout_rate
    )
    
    return model 