"""Multi-Scale Temporal Convolution and Temporal Attention Pooling for EEG."""
import torch
import torch.nn as nn


class MultiScaleTCN(nn.Module):
    """Multi-scale temporal convolution for capturing EEG patterns at different frequencies.

    Different kernel sizes capture different frequency bands:
    - kernel=3: high-frequency gamma (>30Hz)
    - kernel=5: beta (15-30Hz, motion-related V5/MT+)
    - kernel=9: alpha (8-13Hz)
    - kernel=15: theta/delta (<8Hz)
    """
    def __init__(self, dim=2048, kernel_sizes=(3, 5, 9, 15), dropout=0.1):
        super().__init__()
        assert dim % len(kernel_sizes) == 0
        branch_dim = dim // len(kernel_sizes)

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, branch_dim, k, padding=k // 2, groups=1),
                nn.GroupNorm(min(32, branch_dim), branch_dim),
                nn.GELU(),
            ) for k in kernel_sizes
        ])
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """x: (B, S, D) → (B, S, D) with multi-scale temporal features."""
        residual = x
        x_t = x.transpose(1, 2)  # (B, D, S)
        feats = [conv(x_t) for conv in self.convs]
        # All branches should have same length due to padding=k//2
        min_len = min(f.shape[2] for f in feats)
        feats = [f[:, :, :min_len] for f in feats]
        x_out = torch.cat(feats, dim=1).transpose(1, 2)  # (B, S, D)
        x_out = self.proj(x_out)
        # Match sequence length if needed
        if x_out.shape[1] != residual.shape[1]:
            x_out = x_out[:, :residual.shape[1], :]
        return self.norm(residual + x_out)


class TemporalAttentionPool(nn.Module):
    """Learnable query-based attention pooling over temporal/spatial tokens.

    Instead of mean pooling (which discards temporal structure),
    uses a learnable query token + cross-attention to selectively
    aggregate information from the sequence.
    """
    def __init__(self, dim=2048, num_heads=8, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """x: (B, S, D) → (B, D) via attention-weighted pooling."""
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)  # (B, 1, D)
        out, _ = self.attn(q, x, x)  # (B, 1, D)
        out = self.norm(out)
        return out.squeeze(1)  # (B, D)
