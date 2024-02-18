import math

from torch import nn, ones, zeros
import copy


class Encoder(nn.Module):
    # encoder is a stack of N identical layers

    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # for each layer, pass the input and mask
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    # encoder layer includes a multi-head self attention and a feedforward layer, with residual connection

    def __init__(self, h, emb_size, d_ff, dropout):
        super().__init__()
        self.attn = ResBlock(MultiHeadAttention(h, emb_size), emb_size, dropout)
        self.ff = ResBlock(FeedForward(emb_size, d_ff, dropout), emb_size, dropout)

    def forward(self, x, mask):
        return self.ff(self.attn(x, lambda z: (z, z, z, mask)))


class MultiHeadAttention(nn.Module):
    # TODO
    def __init__(self, h, emb_size):
        super().__init__()

    def forward(self, query, key, value):
        pass


class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.fc_2(self.dropout(self.relu(self.fc_1(x))))


class LayerNorm(nn.Module):
    def __init__(self, feat_shape, epsilon=1e-6):
        super().__init__()
        self.w = nn.Parameter(ones(feat_shape))   # learnable parameters
        self.b = nn.Parameter(zeros(feat_shape))  # learnable parameters
        self.epsilon = epsilon  # a minor integer to prevent division by zero

    def forward(self, x):
        mean = x.mean(-1, keepdims=True)
        std = x.std(-1, keepdims=True)
        norm = (x - mean) / (std + self.epsilon)
        return self.w * norm + self.b


class ResBlock(nn.Module):
    # x -> layer_norm -> activation -> dropout -> +x
    def __init__(self, activation, emb_size, dropout):
        super().__init__()
        self.activation = activation
        self.layer_norm = LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.activation(self.layer_norm(x)))


class Generator(nn.Module):
    # also known as LM head
    # linear projection + softmax

    def __init__(self, out_dim, vocab_size):
        super().__init__()
        self.proj = nn.Linear(out_dim, vocab_size)

    def forward(self, x):
        return nn.functional.log_softmax(self.proj(x), dim=-1)  # softmax on the last dimension


class Embedding(nn.Module):
    def __init__(self, emb_dim, vocab_size):
        super().__init__()
        self.emb_layer = nn.Embedding(vocab_size, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, x):
        return self.emb_layer(x) * math.sqrt(self.emb_dim)
