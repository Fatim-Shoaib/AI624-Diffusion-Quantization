"""
=============================================================================
GPTQ Algorithm Implementation
=============================================================================

This is an adaptation of the GPTQ algorithm for SD 3.5 Medium's MMDiT architecture.
Based on the Q-DiT implementation with modifications for diffusion model quantization.

Reference:
- GPTQ Paper: https://arxiv.org/abs/2210.17323
- Q-DiT: https://arxiv.org/abs/2406.17343

The GPTQ algorithm:
1. Collects Hessian information from calibration data
2. Quantizes weights row-by-row using second-order information
3. Compensates for quantization error in remaining weights
"""

import math
import time
import gc
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# Disable TF32 for numerical accuracy during quantization
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class Quantizer_GPTQ(nn.Module):
    """
    Quantizer class for GPTQ that handles scale and zero-point computation.
    
    This quantizer supports:
    - Symmetric and asymmetric quantization
    - Per-channel and per-group quantization
    - MSE-based optimal clipping ratio search
    """
    
    def __init__(self, shape: int = 1):
        """
        Initialize the quantizer.
        
        Args:
            shape: Number of channels (for per-channel quantization)
        """
        super().__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))
        
        # Configuration (set by configure())
        self.bits = None
        self.perchannel = False
        self.sym = True
        self.mse = False
        self.norm = 2.4
        self.grid = 100
        self.maxshrink = 0.8
        self.group_size = -1
        
    def configure(
        self,
        bits: int,
        perchannel: bool = True,
        sym: bool = True,
        mse: bool = False,
        norm: float = 2.4,
        grid: int = 100,
        maxshrink: float = 0.8,
        group_size: int = -1,
    ) -> None:
        """
        Configure the quantizer parameters.
        
        Args:
            bits: Number of bits for quantization (e.g., 4 for W4)
            perchannel: Use per-channel quantization
            sym: Use symmetric quantization
            mse: Use MSE-based optimal clipping
            norm: Lp norm for MSE optimization
            grid: Grid search resolution for MSE
            maxshrink: Maximum shrinkage ratio for MSE search
            group_size: Group size for group-wise quantization (-1 for no grouping)
        """
        self.bits = bits
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.group_size = group_size
        
    def find_params(self, x: torch.Tensor, weight: bool = True) -> None:
        """
        Find optimal scale and zero-point for the given tensor.
        
        Args:
            x: Tensor to find quantization parameters for
            weight: Whether this is a weight tensor (affects reshaping)
        """
        dev = x.device
        self.maxq = self.maxq.to(dev)
        
        shape = x.shape
        
        if self.perchannel:
            if weight:
                # For weights: [out_features, in_features] -> [out_features, -1]
                x = x.flatten(1)
            else:
                # For activations: various shapes
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3]).flatten(1)
                elif len(shape) == 3:
                    x = x.reshape(-1, shape[-1]).t()
                elif len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)
        
        # Find min/max values
        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)
        
        if self.sym:
            # Symmetric quantization
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        
        # Handle zero range
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1
        
        # Compute scale and zero-point
        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)
        
        # MSE-based optimal clipping
        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                
                # Quantize and compute error
                q = self._quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1))
                q = q - x
                q = q.abs().pow(self.norm)
                err = torch.sum(q, 1)
                
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        
        # Reshape scale/zero for weight quantization
        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
        else:
            if len(shape) == 4:
                self.scale = self.scale.reshape(1, -1, 1, 1)
                self.zero = self.zero.reshape(1, -1, 1, 1)
            elif len(shape) == 3:
                self.scale = self.scale.reshape(1, 1, -1)
                self.zero = self.zero.reshape(1, 1, -1)
            elif len(shape) == 2:
                self.scale = self.scale.unsqueeze(0)
                self.zero = self.zero.unsqueeze(0)
    
    def _quantize(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform quantization with given scale and zero-point.
        
        Args:
            x: Input tensor
            scale: Scale factor
            zero: Zero point
            
        Returns:
            Quantized tensor (still in floating point for computation)
        """
        q = torch.clamp(torch.round(x / scale) + zero, 0, self.maxq)
        return scale * (q - zero)
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize the input tensor using computed parameters.
        
        Args:
            x: Input tensor to quantize
            
        Returns:
            Quantized tensor
        """
        if self.ready():
            return self._quantize(x, self.scale, self.zero)
        return x
    
    def enabled(self) -> bool:
        """Check if quantization is enabled."""
        return self.maxq > 0
    
    def ready(self) -> bool:
        """Check if quantization parameters have been computed."""
        return torch.all(self.scale != 0)


class GPTQ:
    """
    GPTQ (Generative Pre-trained Transformer Quantization) implementation.
    
    This class handles the core GPTQ algorithm:
    1. Collect Hessian information from input activations
    2. Quantize weights using second-order optimization
    3. Compensate remaining weights for quantization error
    """
    
    def __init__(self, layer: nn.Linear):
        """
        Initialize GPTQ for a specific layer.
        
        Args:
            layer: The nn.Linear layer to quantize
        """
        self.layer = layer
        self.dev = layer.weight.device
        
        # Get weight shape
        W = layer.weight.data.clone()
        self.rows = W.shape[0]  # out_features
        self.columns = W.shape[1]  # in_features
        
        # Initialize Hessian matrix (H = X^T X)
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        
        # Quantizer will be set before quantization
        self.quantizer: Optional[Quantizer_GPTQ] = None
        
        del W
    
    def add_batch(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        """
        Add a batch of inputs to the Hessian computation.
        
        The Hessian approximation H = X^T X is computed incrementally.
        
        Args:
            inp: Input activations to the layer [batch, seq_len, in_features]
            out: Output of the layer (not used, but kept for hook compatibility)
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        
        # Flatten batch and sequence dimensions
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        
        inp = inp.t()  # [in_features, batch*seq_len]
        
        # Incremental Hessian update: H = (n*H + X^T X) / (n + batch)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
    
    def fasterquant(
        self,
        blocksize: int = 128,
        percdamp: float = 0.01,
        groupsize: int = -1,
        actorder: bool = False,
    ) -> None:
        """
        Perform GPTQ quantization using the collected Hessian.
        
        This implements the efficient GPTQ algorithm with:
        - Lazy batch updates for GPU efficiency
        - Cholesky decomposition for numerical stability
        - Optional group-wise quantization
        
        Args:
            blocksize: Block size for lazy batch updates
            percdamp: Dampening percentage for Hessian diagonal
            groupsize: Group size for group-wise quantization
            actorder: Whether to use activation ordering (not implemented)
        """
        assert not actorder, "Activation ordering not implemented in this version"
        
        W = self.layer.weight.data.clone()
        W = W.float()
        
        tick = time.time()
        
        # Find quantization parameters if not already done
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)
        
        H = self.H.clone()
        del self.H
        
        # Handle dead columns (zero diagonal in Hessian)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        
        # Dampening for numerical stability
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        
        # Cholesky decomposition: H = L L^T
        # Then H^{-1} = L^{-T} L^{-1}
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        
        # Storage for quantized weights and losses
        Q = torch.zeros_like(W)
        Losses = torch.zeros_like(W)
        
        # Process columns in blocks for efficiency
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                
                # Update quantization params for group-wise quantization
                if groupsize > 0:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(
                            W[:, (i1 + i):min((i1 + i + groupsize), self.columns)],
                            weight=True
                        )
                
                # Quantize current column
                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                
                # Compute loss for this column
                Losses1[:, i] = (w - q) ** 2 / d ** 2
                
                # Error compensation: update remaining weights in block
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
            
            # Store results for this block
            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2
            
            # Propagate error to remaining columns
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        
        torch.cuda.synchronize()
        
        # Update layer weights
        Q = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        self.layer.weight.data = Q
        
        # Log statistics
        total_loss = torch.sum(Losses).item()
        logger.debug(f"GPTQ finished in {time.time() - tick:.2f}s, loss: {total_loss:.4f}")
        
        # Cleanup
        del H, Hinv, W, Q, Losses
    
    def free(self) -> None:
        """Free GPU memory used by Hessian."""
        self.H = None
        torch.cuda.empty_cache()
        gc.collect()


def quantize_linear_layer_gptq(
    layer: nn.Linear,
    calibration_inputs: torch.Tensor,
    bits: int = 4,
    groupsize: int = 128,
    percdamp: float = 0.01,
    sym: bool = True,
    mse: bool = False,
) -> nn.Linear:
    """
    Quantize a single linear layer using GPTQ.
    
    Args:
        layer: The linear layer to quantize
        calibration_inputs: Input activations for calibration [num_samples, in_features]
        bits: Number of bits for weight quantization
        groupsize: Group size for group-wise quantization
        percdamp: Dampening percentage
        sym: Use symmetric quantization
        mse: Use MSE-based optimal clipping
        
    Returns:
        Quantized linear layer (in-place modification)
    """
    # Initialize GPTQ
    gptq = GPTQ(layer)
    
    # Configure quantizer
    gptq.quantizer = Quantizer_GPTQ()
    gptq.quantizer.configure(
        bits=bits,
        perchannel=True,
        sym=sym,
        mse=mse,
        group_size=groupsize,
    )
    
    # Add calibration batches
    for i in range(0, calibration_inputs.shape[0], 32):
        batch = calibration_inputs[i:i+32]
        with torch.no_grad():
            out = layer(batch)
        gptq.add_batch(batch, out)
    
    # Perform quantization
    gptq.fasterquant(
        blocksize=128,
        percdamp=percdamp,
        groupsize=groupsize,
    )
    
    # Cleanup
    gptq.free()
    
    return layer
