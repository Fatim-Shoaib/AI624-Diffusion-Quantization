"""
Qronos-DiT: Qronos algorithm adapted for Diffusion Transformers.

This implements the Qronos quantization algorithm from:
"Qronos: Correcting the Past by Shaping the Future in Post-Training Quantization"

Key differences from GPTQ:
1. Collects BOTH X (original activations) AND X̃ (quantized model activations)
2. First iteration uses G = X̃ᵀX and H = X̃ᵀX̃ for better error correction
3. Subsequent iterations use standard GPTQ-style error diffusion

Adapted for diffusion models following Q-DiT's approach.
"""
import math
import time
import gc
from typing import Optional, Tuple, Dict, Any
import warnings

import torch
import torch.nn as nn

from .quant_utils import Quantizer, quantize_gptq, power_iteration


# Disable TF32 for numerical precision
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class QronosDiT:
    """
    Qronos quantization algorithm adapted for Diffusion Transformers.
    
    This class handles the quantization of a single linear layer using the Qronos algorithm.
    It collects both original (X) and quantized (X̃) activations to perform error correction.
    """
    
    def __init__(
        self,
        layer: nn.Linear,
        layer_name: str = "",
    ):
        """
        Initialize Qronos for a layer.
        
        Args:
            layer: The linear layer to quantize
            layer_name: Name of the layer (for logging)
        """
        self.layer = layer
        self.layer_name = layer_name
        self.dev = self.layer.weight.device
        
        # Get weight dimensions
        W = layer.weight.data.clone()
        self.rows = W.shape[0]  # Output features
        self.columns = W.shape[1]  # Input features
        
        # Initialize covariance matrices in float64 for numerical stability
        # H = X̃ᵀX̃ (quantized activations covariance)
        # G = X̃ᵀX (cross-covariance between quantized and original activations)
        self.H = torch.zeros((self.columns, self.columns), device='cpu', dtype=torch.float64)
        self.G = torch.zeros((self.columns, self.columns), device='cpu', dtype=torch.float64)
        
        # Buffer for efficient computation
        self.B = torch.zeros((self.columns, self.columns), device='cpu', dtype=torch.float64)
        
        self.nsamples = 0
        self.n_nonout = W.shape[1]
        
        # Quantizer for this layer
        self.quantizer = Quantizer()
        
        # Buffers for quantized input
        self.quant_input = None
        
        del W
    
    def add_batch_original(self, inp: torch.Tensor):
        """
        Add a batch of ORIGINAL (non-quantized) activations.
        
        This collects X (activations from the original FP model).
        Must be called AFTER add_batch_quantized for each batch.
        
        Args:
            inp: Input activations from original model [batch, seq_len, features]
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        
        batch_size = inp.shape[0]
        
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()  # [features, batch*seq_len]
        
        inp = inp.to(torch.float64).to('cpu')
        
        # Update G = X̃ᵀX using the stored quantized input
        if self.quant_input is not None:
            # G += X̃ᵀX (normalized incrementally)
            self.B.copy_(self.quant_input.matmul(inp.t()))
            self.G *= (self.nsamples - batch_size) / self.nsamples
            self.G += self.B / self.nsamples
            self.quant_input = None
    
    def add_batch_quantized(self, inp: torch.Tensor):
        """
        Add a batch of QUANTIZED activations (from partially quantized model).
        
        This collects X̃ (activations from the quantized model).
        Must be called BEFORE add_batch_original for each batch.
        
        Args:
            inp: Input activations from quantized model [batch, seq_len, features]
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        
        batch_size = inp.shape[0]
        
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()  # [features, batch*seq_len]
        
        inp = inp.to(torch.float64).to('cpu')
        
        # Update sample count
        self.nsamples += batch_size
        
        # Update H = X̃ᵀX̃
        self.B.copy_(inp.matmul(inp.t()))
        self.H *= (self.nsamples - batch_size) / self.nsamples
        self.H += self.B / self.nsamples
        
        # Store quantized input for G computation
        self.quant_input = inp.clone()
    
    def add_batch(self, inp: torch.Tensor, out: torch.Tensor = None):
        """
        Legacy interface: Add a batch of activations.
        
        For backward compatibility with GPTQ-style interface.
        This is used when X = X̃ (no distinction between original and quantized).
        
        Args:
            inp: Input activations
            out: Output activations (unused, for compatibility)
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        
        batch_size = inp.shape[0]
        
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        
        # Use float64 for stability
        inp = inp.to(torch.float64).to('cpu')
        
        # Update H incrementally
        self.H *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size
        inp_scaled = math.sqrt(2 / self.nsamples) * inp
        self.H += inp_scaled.matmul(inp_scaled.t())
        
        # When G is not separately set, use H (X = X̃ case)
        self.G = self.H.clone()
    
    def fasterquant(
        self,
        blocksize: int = 128,
        percdamp: float = 0.01,
        groupsize: int = -1,
        actorder: bool = False,
        alpha: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Perform Qronos quantization on the layer.
        
        This implements Algorithm 1 from the Qronos paper:
        1. First iteration: Use both G and H for error correction
        2. Subsequent iterations: GPTQ-style error diffusion via Cholesky
        
        Args:
            blocksize: Block size for lazy batch updates
            percdamp: Percentage damping for Hessian diagonal
            groupsize: Group size for quantization (-1 for per-channel)
            actorder: Whether to order by activation magnitude (not recommended for Qronos)
            alpha: Damping factor based on max singular value (Qronos-specific)
        
        Returns:
            Dictionary with quantization statistics
        """
        assert not actorder, "actorder is not recommended for Qronos"
        
        # Get weight matrix
        W = self.layer.weight.data.clone()
        W = W.float()  # Work in float32 for weight operations
        W_orig = W.clone()
        
        tick = time.time()
        
        # Setup quantizer if not ready
        if not self.quantizer.ready():
            self.quantizer.find_params(W[:, :self.n_nonout], weight=True)
        
        # Move covariance matrices to computation device
        H = self.H.clone().to(self.dev)
        G = self.G.clone().to(self.dev)
        
        del self.H, self.G, self.B
        
        # Handle dead columns (zero diagonal)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        
        # Initialize quantized weight matrix
        Q = torch.zeros_like(W)
        Losses = torch.zeros_like(W)
        
        # === QRONOS FIRST ITERATION ===
        # q_1 = Q((G[1,≥1] @ w - H[1,≥2] @ w[≥2]) / H[1,1])
        # w_≥2 = H[≥2,≥2]^{-1} @ (G[≥2,≥1] @ w - H[≥2,1] @ q_1)
        
        # Compute damped inverse Hessian using power iteration for damping
        try:
            # Use alpha * sigma_max for damping (Qronos-specific)
            sigma_max = power_iteration(H.to(torch.float64), num_iters=30)
            damp = alpha * sigma_max
        except:
            # Fallback to percentage damping
            damp = percdamp * torch.mean(torch.diag(H)).item()
        
        # Add damping
        diag_idx = torch.arange(self.columns, device=self.dev)
        H_damped = H.to(torch.float64).clone()
        H_damped[diag_idx, diag_idx] += damp
        
        G = G.to(torch.float64)
        
        # Compute inverse Hessian via Cholesky
        try:
            L = torch.linalg.cholesky(H_damped)
            H_inv = torch.cholesky_inverse(L)
            # Cholesky of inverse for error diffusion (with stabilization constant)
            c = 1e4
            L_inv = torch.linalg.cholesky(H_inv * c, upper=True) / math.sqrt(c)
        except Exception as e:
            warnings.warn(f"Cholesky decomposition failed for {self.layer_name}: {e}. Using pseudo-inverse.")
            H_inv = torch.linalg.pinv(H_damped)
            L_inv = torch.linalg.cholesky(H_inv + 1e-6 * torch.eye(self.columns, device=self.dev, dtype=torch.float64), upper=True)
        
        # Process each output channel
        for row_idx in range(self.rows):
            w = W[row_idx, :].to(torch.float64)
            w_orig = W_orig[row_idx, :].to(torch.float64)
            
            # === First iteration (t=1): Qronos error correction ===
            # q_1 = Q((G[0,≥0] @ w_orig - H[0,≥1] @ w[≥1]) / H[0,0])
            numerator = G[0, :] @ w_orig - H[0, 1:] @ w[1:]
            denominator = H[0, 0]
            
            if denominator.abs() < 1e-10:
                q_1 = w[0]
            else:
                q_1_arg = (numerator / denominator).float()
                q_1 = quantize_gptq(
                    q_1_arg.unsqueeze(0).unsqueeze(1),
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq,
                    self.quantizer.channel_group,
                ).flatten()[0].to(torch.float64)
            
            Q[row_idx, 0] = q_1.float()
            Losses[row_idx, 0] = ((w[0] - q_1) ** 2).float()
            
            # === Update remaining weights after first iteration ===
            # w_≥2 = H[≥1,≥1]^{-1} @ (G[≥1,≥0] @ w_orig - H[≥1,0] @ q_1)
            if self.columns > 1:
                rhs = G[1:, :] @ w_orig - H[1:, 0] * q_1
                try:
                    w_updated = torch.linalg.solve(H_damped[1:, 1:], rhs)
                except:
                    w_updated = H_inv[1:, 1:] @ rhs
                
                w[1:] = w_updated
            
            # === Subsequent iterations (t≥2): GPTQ-style error diffusion ===
            for i1 in range(1, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1
                
                # Get block of weights
                W1 = w[i1:i2].clone()
                Q1 = torch.zeros(count, dtype=torch.float64, device=self.dev)
                Err1 = torch.zeros(count, dtype=torch.float64, device=self.dev)
                Losses1 = torch.zeros(count, dtype=torch.float64, device=self.dev)
                
                # Get inverse Hessian block (adjusted for Qronos: t-1 indexing)
                # After Sherman-Morrison update, we use L_inv starting from index 1
                Hinv1 = L_inv[i1-1:i2-1, i1-1:i2-1] if i1 > 0 else L_inv[i1:i2, i1:i2]
                
                for i in range(count):
                    # Quantize current weight (RTN)
                    col_idx = i1 + i
                    w_val = W1[i].float()
                    
                    # Handle groupsize
                    if groupsize > 0 and col_idx % groupsize == 0:
                        self.quantizer.find_params(
                            W[row_idx, col_idx:min(col_idx + groupsize, self.n_nonout)].unsqueeze(0),
                            weight=True
                        )
                    
                    q_val = quantize_gptq(
                        w_val.unsqueeze(0).unsqueeze(1),
                        self.quantizer.scale,
                        self.quantizer.zero,
                        self.quantizer.maxq,
                        self.quantizer.channel_group,
                    ).flatten()[0].to(torch.float64)
                    
                    Q1[i] = q_val
                    
                    # Compute error
                    d = Hinv1[i, i]
                    if d.abs() < 1e-10:
                        d = torch.tensor(1.0, dtype=torch.float64, device=self.dev)
                    
                    err = (W1[i] - q_val) / d
                    Err1[i] = err
                    Losses1[i] = (W1[i] - q_val) ** 2 / d ** 2
                    
                    # Update remaining weights in block
                    W1[i:] -= err * Hinv1[i, i:].to(W1.dtype)
                
                # Store results
                Q[row_idx, i1:i2] = Q1.float()
                Losses[row_idx, i1:i2] = Losses1.float() / 2
                
                # Update remaining weights outside block
                if i2 < self.columns:
                    w[i2:] -= Err1 @ L_inv[i1-1:i2-1, i2-1:].to(w.dtype)
        
        torch.cuda.synchronize()
        elapsed = time.time() - tick
        total_loss = torch.sum(Losses).item()
        
        # Update layer weights
        self.layer.weight.data = Q.to(self.layer.weight.dtype)
        
        # Cleanup
        del H, G, H_inv, L_inv, H_damped
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            'time': elapsed,
            'loss': total_loss,
            'layer_name': self.layer_name,
        }
    
    def free(self):
        """Free memory used by the quantizer."""
        self.H = None
        self.G = None
        self.B = None
        self.quant_input = None
        gc.collect()
        torch.cuda.empty_cache()


class QronosDiTSimple:
    """
    Simplified Qronos implementation for cases where X = X̃.
    
    This is essentially GPTQ with the Qronos damping strategy.
    Used when we don't have separate original and quantized activations.
    """
    
    def __init__(self, layer: nn.Linear, layer_name: str = ""):
        self.layer = layer
        self.layer_name = layer_name
        self.dev = self.layer.weight.device
        
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        
        # Single Hessian matrix (X = X̃ case)
        self.H = torch.zeros((self.columns, self.columns), device='cpu', dtype=torch.float64)
        self.nsamples = 0
        self.n_nonout = W.shape[1]
        
        self.quantizer = Quantizer()
        del W
    
    def add_batch(self, inp: torch.Tensor, out: torch.Tensor = None):
        """Add a batch of activations."""
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        
        batch_size = inp.shape[0]
        
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        
        inp = inp.to(torch.float64).to('cpu')
        
        self.H *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size
        inp_scaled = math.sqrt(2 / self.nsamples) * inp
        self.H += inp_scaled.matmul(inp_scaled.t())
    
    def fasterquant(
        self,
        blocksize: int = 128,
        percdamp: float = 0.01,
        groupsize: int = -1,
        actorder: bool = False,
        alpha: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Perform quantization (simplified Qronos/GPTQ hybrid).
        """
        W = self.layer.weight.data.clone().float()
        
        tick = time.time()
        
        if not self.quantizer.ready():
            self.quantizer.find_params(W[:, :self.n_nonout], weight=True)
        
        H = self.H.clone().to(self.dev)
        del self.H
        
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        
        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        
        # Damping with power iteration
        try:
            sigma_max = power_iteration(H, num_iters=30)
            damp = alpha * sigma_max
        except:
            damp = percdamp * torch.mean(torch.diag(H)).item()
        
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        
        # Cholesky decomposition
        try:
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            c = 1e4
            H = torch.linalg.cholesky(H * c, upper=True) / math.sqrt(c)
            Hinv = H
        except Exception as e:
            warnings.warn(f"Cholesky failed: {e}")
            return {'time': 0, 'loss': float('inf'), 'layer_name': self.layer_name}
        
        # Standard GPTQ quantization loop
        for i1 in range(0, self.n_nonout, blocksize):
            i2 = min(i1 + blocksize, self.n_nonout)
            count = i2 - i1
            
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                
                if groupsize > 0:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(
                            W[:, (i1 + i):min((i1 + i + groupsize), self.n_nonout)],
                            weight=True
                        )
                
                q = quantize_gptq(
                    w.unsqueeze(1),
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq,
                    self.quantizer.channel_group,
                ).flatten()
                
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2
                
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
            
            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        
        torch.cuda.synchronize()
        elapsed = time.time() - tick
        total_loss = torch.sum(Losses).item()
        
        self.layer.weight.data = Q.to(self.layer.weight.dtype)
        
        del H, Hinv
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            'time': elapsed,
            'loss': total_loss,
            'layer_name': self.layer_name,
        }
    
    def free(self):
        self.H = None
        gc.collect()
        torch.cuda.empty_cache()
