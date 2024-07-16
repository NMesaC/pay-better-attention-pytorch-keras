"""
## "You Need to Pay Better Attention" PyTorch Implementation

## Paper Link: https://arxiv.org/abs/2403.01643

## Author: Nicholas Mesa-Cucalon (https://github.com/NMesaC)

## NOTE: This implementation has a W_a matrix FOR EACH attention layer
"""
import math
import torch
from torch import nn

class AttentionLayer(nn.Module):
    def __init__(self, 
                 d_model : int, 
                 d_q : int, 
                 d_k : int, 
                 d_v : int,
                 layer_type : str = 'SDPA',
                 idx : int = 0,
                 max_len : int = 32):
        super().__init__()
        self.d_model    = d_model 
        self.d_q        = d_q
        self.d_k        = d_k
        self.d_v        = d_v
        self.layer_type = layer_type
        self.idx        = idx
        self.max_len    = max_len
        self._set_layer_type()


    def _set_layer_type(self):
        self.softmax = nn.Softmax(dim = 1)
        self.W_q     = nn.Linear(self.d_model,self.d_q)
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.constant_(self.W_q.bias, 0)
        if self.layer_type == 'Optimised':
            self.W_k     = nn.Linear(self.d_model,self.d_k)
            nn.init.xavier_uniform_(self.W_k.weight)
            nn.init.constant_(self.W_k.bias, 0)
            self.forward = self._forward_optimised
        elif self.layer_type == 'Efficient':
            self.forward = self._forward_efficient
        elif self.layer_type == 'Super':
            self.forward = self._forward_super
            self.W_a     = nn.Linear(self.max_len,self.max_len)
            nn.init.xavier_uniform_(self.W_a.weight)
            nn.init.constant_(self.W_a.bias, 0)
        else:
            # Default to SDPA
            self.W_k     = nn.Linear(self.d_model,self.d_k)
            self.W_v     = nn.Linear(self.d_model,self.d_v)
            nn.init.xavier_uniform_(self.W_k.weight)
            nn.init.constant_(self.W_k.bias, 0)
            nn.init.xavier_uniform_(self.W_v.weight)
            nn.init.constant_(self.W_v.bias, 0)
            self.forward = self._forward_SDPA

    def _forward_SDPA(self, inp_q, inp_k, inp_v):
        Q     = self.W_q(inp_q)
        K     = self.W_k(inp_k)
        V     = self.W_v(inp_v)
        K_t   = K.permute(0,2,1)
        S     = self.softmax((Q @ K_t) / math.sqrt(self.d_q))
        H     = S @ V
        return H
        
    def _forward_optimised(self, inp_q : torch.Tensor, inp_k : torch.Tensor, inp_v : torch.Tensor):
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
        Q     = self.W_q(inp_q)
        lo    = ((self.idx) * self.d_k)
        hi    = ((self.idx + 1) * self.d_k)
        K_t   = inp_k[:, :, lo : hi].permute(0,2,1)
        S     = self.softmax((Q @ K_t) / math.sqrt(self.d_q))
        V     = inp_v[:,:, lo : hi]
        H     = S @ V
        return H

    def _forward_super(self, inp_q : torch.Tensor, inp_k : torch.Tensor, inp_v : torch.Tensor):
        Q     = self.W_q(inp_q)
        lo    = ((self.idx) * self.d_k)
        hi    = ((self.idx + 1) * self.d_k)
        K_t   = inp_k[:, :, lo : hi].permute(0,2,1)
        S     = self.softmax((Q @ K_t) / math.sqrt(self.d_q))
        V     = self.W_a(inp_v[:,:, lo : hi].permute(0,2,1)).permute(0,2,1)
        H     = S @ V
        return H
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, max_len, layer_type):
        super().__init__()
        self.layers  = nn.Sequential()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k     = d_k
        self.d_v     = d_v
        for i in range(n_heads):
            self.layers.add_module("Attention_Layer "+str(i),
                                   AttentionLayer(d_model,d_k,d_k,d_v,layer_type,i,max_len))
        self.W_o     = nn.Linear(n_heads * d_v, d_model)

    def forward(self, inp_q, inp_k, inp_v):
        for i, layer in enumerate(self.layers):
            if i == 0:
                H = layer(inp_q,inp_k,inp_v)
            else:
                h_i = layer(inp_q,inp_k,inp_v)
                h_cat = (H.clone(),h_i)
                H = torch.cat(h_cat,2)
        out = self.W_o(H)
        return out