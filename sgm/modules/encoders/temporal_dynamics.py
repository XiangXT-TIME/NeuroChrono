"""P1: Temporal Dynamics Decoder - bf16-safe version.

Cross-attention decoder: T_out+1 query tokens (T_out temporal + 1 global)
selectively aggregate temporal info from EEG 226 spatial tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BF16SafeAttention(nn.Module):
    """Manual multi-head attention, avoids nn.MultiheadAttention bf16 CUDA bugs."""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, causal_mask=None):
        B, Sq, D = q.shape
        Sk = k.shape[1]
        H, Dh = self.nhead, self.head_dim
        q = self.q_proj(q).reshape(B, Sq, H, Dh).transpose(1, 2)
        k = self.k_proj(k).reshape(B, Sk, H, Dh).transpose(1, 2)
        v = self.v_proj(v).reshape(B, Sk, H, Dh).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if causal_mask is not None:
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, Sq, D)
        return self.out_proj(out)


class DecoderLayer(nn.Module):
    """Single decoder layer: cross-attn + self-attn + FFN. bf16-safe."""
    def __init__(self, d_model, nhead, dim_ff, dropout=0.1):
        super().__init__()
        self.cross_attn = BF16SafeAttention(d_model, nhead, dropout)
        self.self_attn = BF16SafeAttention(d_model, nhead, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, memory, causal_mask=None):
        q = self.norm1(queries)
        h = self.cross_attn(q, memory, memory)
        queries = queries + self.dropout(h)
        q = self.norm2(queries)
        h = self.self_attn(q, q, q, causal_mask=causal_mask)
        queries = queries + self.dropout(h)
        queries = queries + self.ffn(self.norm3(queries))
        return queries


class TemporalDynamicsDecoder(nn.Module):
    def __init__(self, input_dim=2048, d_model=512, nhead=8, num_layers=4,
                 t_out=9, out_dim=1152, dropout=0.1, use_causal_mask=False):
        super().__init__()
        self.t_out = t_out
        self.d_model = d_model
        self.use_causal_mask = use_causal_mask

        self.temporal_queries = nn.Parameter(torch.randn(1, t_out, d_model) * 0.02)
        self.global_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.input_proj = nn.Linear(input_dim, d_model)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.temporal_out_proj = nn.Linear(d_model, out_dim)
        self.global_out_proj = nn.Linear(d_model, out_dim)

        self.flow_traj_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, eeg_spatial):
        B = eeg_spatial.shape[0]
        memory = self.input_proj(eeg_spatial)

        t_queries = self.temporal_queries.expand(B, -1, -1)
        g_query = self.global_query.expand(B, -1, -1)
        queries = torch.cat([t_queries, g_query], dim=1)

        # Causal mask for self-attention: temporal queries can only attend to earlier ones
        causal_mask = None
        if self.use_causal_mask:
            S = queries.shape[1]  # T_out + 1
            causal_mask = torch.triu(torch.ones(S, S, device=queries.device, dtype=torch.bool), diagonal=1)

        for layer in self.decoder_layers:
            queries = layer(queries, memory, causal_mask=causal_mask)
        queries = self.norm(queries)

        t_out = queries[:, :self.t_out, :]
        g_out = queries[:, self.t_out, :]

        temporal_tokens = self.temporal_out_proj(t_out)
        global_dyn_token = self.global_out_proj(g_out)
        flow_traj_pred = self.flow_traj_head(t_out).squeeze(-1)

        return temporal_tokens, global_dyn_token, flow_traj_pred
