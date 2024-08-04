import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab, n_embed, n_token):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embed)

        # A learnable weight matrix encodes the position information for each token
        self.position_embedding = nn.parameters(torch.zeros((n_token, n_embed)))
    
    def forward(self, token):
        #batch_size, seq_len -> batch_size, seq_len, dim
        x = self.token_embedding(token)

        x += self.position_embedding
        return x
    
class CLIPLayer(nn.Module):
    def __init__(self, n_head, n_embed):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embed)

        self.attention = SelfAttention(n_heads=n_head, d_embed=n_embed)

        self.layernorm_2 = nn.LayerNorm(n_embed)

        #feed forward layer
        self.fc_1 = nn.Linear(n_embed, 4 * n_embed)
        self.fc_2 = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x):
        #shape of x: (batch_size, seq_len, dim)

        residue = x

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_1(x)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.attention(x, causal_mask=True)

        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        #apply feedforward layer
        residue = x

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_2(x)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = self.fc_1(x)

        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = x * torch.sigmoid(1.702 * x)

        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.fc_2(x)

        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        return x


    







class CLIP(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.ModuleList(
            [
                CLIPLayer(12, 768) for i in range(12)
            ]
        )

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens):
        tokens = tokens.type(torch.long)

        #batch_size, seq_len -> batch_size, seq_len, dim
        state = self.embedding(tokens)

        # Apply encoder layers similar to the Transformer's encoder.
        for layer in self.layers:
            #batch_size, seq_len, dim -> batch_size, seq_len, dim
            state = layer(state)
        #batch_size, seq_len, dim -> batch_size, seq_len, dim
        output = self.layernorm(state)

        return output
