"""Common Transformer building blocks shared across encoders."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention."""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, attn_mask=None):
        B, S, _ = x.shape
        q = self.q_proj(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, self.embed_dim)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """Cross-attention: query attends to key/value from another modality."""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, context):
        B, Sq, _ = query.shape
        Sc = context.shape[1]
        q = self.q_proj(query).reshape(B, Sq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).reshape(B, Sc, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).reshape(B, Sc, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, Sq, self.embed_dim)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(self.norm1(src), src_mask)
        src = src + self.dropout1(src2)
        src2 = self.feed_forward(self.norm2(src))
        src = src + self.dropout2(src2)
        return src
