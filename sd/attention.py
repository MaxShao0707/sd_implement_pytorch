import torch
import math
import torch.nn as nn
import torch.nn.functional.F as F

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        #combins wq, wk, wv into one matrix
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias = in_proj_bias)

        #wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // self.n_heads

    def forward(self, x, causal_mask=False):
        #x: batch_size, seq_len, dim
        
        #(batch_size, seq_len, dim)
        input_shape = x.shape

        batch_size, seq_len, dim = input_shape

        #batch_size, seq_len, H, dim / H
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        #batch_size, seq_len, dim * 3 -> 3 tensor of batch_size, seq_len, dim
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        #(batch_size, seq_len, dim) -> (batch_size, seq_len, h, dim / h) -> (batch_size, h, seq_len, dim / h)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)
        
        #(batch_size, h, seq_len, dim/h) @ (batch_size, h, dim / h, seq_len) -> (batch_size, h, seq_len, seq_len)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(weight, dtype=bool).triu(1)
            # Fill the upper triangle with -inf
            weight.masked_fill_(mask, -torch.inf)

        # Divide by d_k (Dim / H).
        # batch_size, h, seq_len, seq_len -> batch_size, h, seq_len, seq_len
        weight /= math.sqrt(self.d_head)

        #batch_size, h, seq_len, seq_len -> batch_size, h, seq_len, seq_len
        weight = F.softmax(weight, dim=-1)

        #batch_size, h, seq_len, seq_len @ batch_size, h, seq_len, dim / h ->batch_size, h, seq_len, dim / h
        output = weight @ v

        #batch_size, h, seq_len, dim / h -> batch_size, seq_len, h, dim / h
        output = output.transpose(1, 2)

        #batch_size, seq_len, h, dim / h -> batch_size, seq_len, dim
        output = output.reshape(input_shape)

        #batch_size, seq_len, dim -> batch_size, seq_len, dim
        output = self.out_proj(output)
        #batch_size, seq_len, dim
        return output
    


