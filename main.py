import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from caffe2.python.helpers.dropout import dropout
from torch.nn.functional import linear


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, src, trg):
        return


class Generator(nn.Module):
    def __init__(self, d_model, d_vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, d_vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

"""
Encoder 和 Decoder
"""
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.dropout = dropout
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, tgt_mask, src_mask):
        self.sublayer[0](x, lambda x: self.self_attn(x, x, x, src_mask))
        self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, tgt_mask))
        return self.sublayer[2](x, self.feed_forward)


# def subsequent_mask(size):
#     attn_type = (1, size, size)
#     subsequent_mask = torch.triu(torch.ones(attn_type), 1).type(torch.uint8)
#     return subsequent_mask == 0

"""
Attention
"""
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(score, dim=-1)  # 函数式编程写法
    # score = score.softmax(dim=-1) # 两种写法，张量的面向对象写法
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MutiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.h = h
        self.d_k = d_model // h

    def forward(self, query, key, value, mask=None):
        nbatchs = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)
        # x通过线性层得到 q k v
        query, key, value = [linear(x).view(nbatchs, -1, self.h, self.d_k).transpose(1, 2) for linear, x in zip(self.linears, (query, key, value))]
        # self-attention -> (batch, num_head, seq_len, d_k)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)   # x -> (batch, num_head, seq_len, d_k)
        # concat -> (batch, seq_len, num_head * d_k)
        x = (x.transpose(1, 2)
             .contiguous()
             .view(nbatchs, -1, self.h * self.d_k))
        del query, key, value
        return self.linears[-1](x)

"""
Feed Forward Network
"""

