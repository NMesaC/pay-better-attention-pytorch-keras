import math
import torch
import numpy as np
from torch import nn

class AttentionLayer(nn.Module):
    def __init__(self, 
                 d_model : int, 
                 d_q : int, 
                 d_k : int, 
                 d_v : int,
                 layer_type : str = 'SDPA',
                 idx : int = 0,
                 max_len : int = 50):
        super().__init__()
        self.d_model    = d_model # Embedding len
        self.d_q        = d_q
        self.d_k        = d_k
        self.d_v        = d_v
        self.layer_type = layer_type
        self.idx        = idx
        self.max_len    = max_len
        self._set_layer_type()

    def _set_layer_type(self):
        # General layers and functions
        self.softmax    = nn.Softmax(dim = 1)
        self.W_q     = nn.Linear(self.d_model,self.d_q)
        if self.layer_type == 'Optimised':
            self.W_k     = nn.Linear(self.d_model,self.d_k)
            self.forward = self._forward_optimised
        elif self.layer_type == 'Efficient':
            self.forward = self._forward_efficient
        elif self.layer_type == 'Super':
            self.W_a     = nn.Linear(self.max_len,self.max_len)
            self.forward = self._forward_super
        else:
            # Default to SDPA
            self.W_k     = nn.Linear(self.d_model,self.d_k)
            self.W_v     = nn.Linear(self.d_model,self.d_v)
            self.forward = self._forward_SDPA

    def _forward_SDPA(self, inp_q, inp_k, inp_v):
        # Scaled Dot Product Attention Forward Pass
        Q     = self.W_q(inp_q)
        K     = self.W_k(inp_k)
        V     = self.W_v(inp_v)
        K_t   = K.permute(0,2,1) #Transpose along actual dims, not batch dims
        S     = self.softmax((Q @ K_t) / math.sqrt(self.d_q))
        H     = S @ V
        return H
        
    def _forward_optimised(self, inp_q : torch.Tensor, inp_k : torch.Tensor, inp_v : torch.Tensor):
        # Optimized Attention Forward Pass
        # NOTE: Assumes d_v = d_m / h
        Q     = self.W_q(inp_q)
        K     = self.W_k(inp_k)
        K_t   = K.permute(0,2,1)
        S     = self.softmax((Q @ K_t) / math.sqrt(self.d_q))
        v_lo  = ((self.idx) * self.d_v)
        v_hi  = ((self.idx + 1) * self.d_v)
        V     = inp_v[:,:, v_lo : v_hi]
        H     = S @ V
        return H

    def _forward_efficient(self, inp_q : torch.Tensor, inp_k : torch.Tensor, inp_v : torch.Tensor):
        # Efficient Attention Forward Pass
        # NOTE: Assumes d_v = d_m / h and d_k = d_m / h => d_v = d_k
        Q     = self.W_q(inp_q)
        lo    = ((self.idx) * self.d_k)
        hi    = ((self.idx + 1) * self.d_k)
        K_t   = inp_k[:, :, lo : hi].permute(0,2,1)
        S     = self.softmax((Q @ K_t) / math.sqrt(self.d_q))
        V     = inp_v[:,:, lo : hi]
        H     = S @ V
        return H

    def _forward_super(self, inp_q : torch.Tensor, inp_k : torch.Tensor, inp_v : torch.Tensor):
        # Super Attention Forward Pass
        # NOTE: Assumes d_v = d_m / h and d_k = d_m / h => d_v = d_k
        Q     = self.W_q(inp_q)
        lo    = ((self.idx) * self.d_k)
        hi    = ((self.idx + 1) * self.d_k)
        K_t   = inp_k[:, :, lo : hi].permute(0,2,1)
        S     = self.softmax((Q @ K_t) / math.sqrt(self.d_q))
        V     = self.W_a.weight @ (inp_v[:,:, lo : hi])
        H     = S @ V
        return H
