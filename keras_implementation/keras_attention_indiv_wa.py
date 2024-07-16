"""
## "You Need to Pay Better Attention" Keras Implementation

## Paper Link: https://arxiv.org/abs/2403.01643

## Author: Nicholas Mesa-Cucalon (https://github.com/NMesaC)

## NOTE: This implementation has a W_a matrix FOR EACH attention layer
"""
import keras
import tensorflow as tf
from keras import ops
from keras import layers

class AttentionLayer(keras.layers.Layer):
    def __init__(self,
                 d_model: int,
                 d_q: int,
                 d_k: int,
                 d_v: int,
                 layer_type: str = 'SDPA',
                 idx: int = 0,
                 max_len: int = 32,
                 **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.layer_type = layer_type
        self.idx = idx
        self.max_len = max_len

    def build(self, input_shape):
        self.W_q = layers.Dense(self.d_q)
        self.W_q.build((None, self.d_model))
        if self.layer_type in ['SDPA', 'Optimised']:
            self.W_k = layers.Dense(self.d_k)
            self.W_k.build((None, self.d_model))
        if self.layer_type == 'SDPA':
            self.W_v = layers.Dense(self.d_v)
            self.W_v.build((None, self.d_model))
        if self.layer_type == 'Super':
            self.W_a = layers.Dense(self.max_len)
            self.W_a.build((None, self.max_len))

        super().build(input_shape)

    def call(self, inputs):
        inp_q, inp_k, inp_v = inputs
        if self.layer_type == 'Optimised':
            return self._forward_optimised(inp_q, inp_k, inp_v)
        elif self.layer_type == 'Efficient':
            return self._forward_efficient(inp_q, inp_k, inp_v)
        elif self.layer_type == 'Super':
            return self._forward_super(inp_q, inp_k, inp_v)
        else:
            return self._forward_SDPA(inp_q, inp_k, inp_v)

    def _forward_SDPA(self, inp_q, inp_k, inp_v):
        Q = self.W_q(inp_q)
        K = self.W_k(inp_k)
        V = self.W_v(inp_v)
        K_t = tf.transpose(K, perm=[0, 2, 1])
        S = tf.nn.softmax((Q @ K_t) / tf.math.sqrt(tf.cast(self.d_q, tf.float32)), axis=1)
        H = S @ V
        return H

    def _forward_optimised(self, inp_q, inp_k, inp_v):
        Q = self.W_q(inp_q)
        K = self.W_k(inp_k)
        K_t = tf.transpose(K, perm=[0, 2, 1])
        S = tf.nn.softmax((Q @ K_t) / tf.math.sqrt(tf.cast(self.d_q, tf.float32)), axis=1)
        v_lo = self.idx * self.d_v
        v_hi = (self.idx + 1) * self.d_v
        V = inp_v[:, :, v_lo:v_hi]
        H = S @ V
        return H

    def _forward_efficient(self, inp_q, inp_k, inp_v):
        Q = self.W_q(inp_q)
        lo = self.idx * self.d_k
        hi = (self.idx + 1) * self.d_k
        K_t = tf.transpose(inp_k[:, :, lo:hi], perm=[0, 2, 1])
        S = tf.nn.softmax((Q @ K_t) / tf.math.sqrt(tf.cast(self.d_q, tf.float32)), axis=1)
        V = inp_v[:, :, lo:hi]
        H = S @ V
        return H

    def _forward_super(self, inp_q, inp_k, inp_v):
        Q = self.W_q(inp_q)
        lo = self.idx * self.d_k
        hi = (self.idx + 1) * self.d_k
        K_t = tf.transpose(inp_k[:, :, lo:hi], perm=[0, 2, 1])
        S = tf.nn.softmax((Q @ K_t) / tf.math.sqrt(tf.cast(self.d_q, tf.float32)), axis=1)
        V = self.W_a.kernel @ inp_v[:, :, lo:hi]
        H = S @ V
        return H

class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, n_heads, d_model, d_k, d_v, max_len, layer_type, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.max_len = max_len
        self.layer_type = layer_type

    def build(self,input_shape):
        self.attention_layers = [
            AttentionLayer(d_model=self.d_model, 
                           d_q=self.d_k,
                           d_k=self.d_k,
                           d_v=self.d_v,
                           layer_type=self.layer_type, 
                           idx=i, 
                           max_len=self.max_len)
            for i in range(self.n_heads)
        ]

        # Build each attention layer
        for layer in self.attention_layers:
            layer.build(input_shape)

        # Build the output dense layer
        self.W_o = layers.Dense(self.d_model)
        self.W_o.build((None, self.n_heads * self.d_v))

        super().build(input_shape)

    def call(self, inputs):
        inp_q, inp_k, inp_v = inputs, inputs, inputs

        H = None
        for i, layer in enumerate(self.attention_layers):
            h_i = layer([inp_q, inp_k, inp_v])
            if i == 0:
                H = h_i
            else:
                H = tf.concat([H, h_i], axis=-1)

        out = self.W_o(H)
        return out