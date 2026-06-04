"""Fast Branch v2: EEG → fMRI Feature Distillation.

Instead of classification heads (dynamics/direction/motion/TC),
the fast branch learns to align EEG features with fMRI features
via MSE distillation. This leverages the fMRI encoder (trained in Stage 1A)
as a teacher to guide the EEG encoder.
"""
import torch
import torch.nn as nn
from sgm.modules.encoders.temporal_conv import TemporalAttentionPool


class DistillationProjector(nn.Module):
    """Project EEG features to match fMRI feature space for distillation."""
    def __init__(self, in_dim=2048, out_dim=2048):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        """x: (B, D) pooled EEG features → (B, out_dim) projected features."""
        return self.proj(x)


class FastBranch(nn.Module):
    """
    Fast Branch v2: EEG encoder + distillation projector.

    Instead of 4 classification/regression heads, uses a single projector
    to map EEG features toward fMRI feature space. The distillation loss
    (MSE between projected EEG and detached fMRI features) provides the
    training signal.

    P1 extension: optional TemporalDynamicsDecoder for multi-frame temporal
    dynamics modeling (cross-attention decoder from EEG spatial tokens).
    """
    def __init__(
        self,
        eeg_encoder,
        embed_dim=2048,
        head_dim=1152,
        # Legacy params (ignored, kept for config compatibility)
        use_dynamics_head=True,
        use_motion_head=True,
        use_temporal_coherence_head=True,
        use_direction_head=True,
        motion_pca_dim=128,
        num_dyn_classes=3,
        num_dir_classes=8,
        # P1: Temporal Dynamics Decoder params
        use_temporal_dynamics=False,
        num_temporal_queries=9,
        temporal_d_model=512,
        use_causal_mask=False,
    ):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.use_temporal_dynamics = use_temporal_dynamics

        # Temporal attention pooling for sequence → vector
        self.temporal_pool = TemporalAttentionPool(dim=embed_dim)

        # Distillation projector: map pooled EEG to fMRI-compatible space
        # Spatial-level: (B, 226, 2048) → (B, 226, 2048)
        self.spatial_projector = DistillationProjector(embed_dim, embed_dim)
        # CLS-level: (B, 1152) → (B, 1152) for CLS token alignment
        self.cls_projector = DistillationProjector(head_dim, head_dim)

        # P1: Temporal Dynamics Decoder
        if use_temporal_dynamics:
            from sgm.modules.encoders.temporal_dynamics import TemporalDynamicsDecoder
            self.temporal_decoder = TemporalDynamicsDecoder(
                input_dim=embed_dim,
                d_model=temporal_d_model,
                nhead=8,
                num_layers=4,
                t_out=num_temporal_queries,
                out_dim=head_dim,
                dropout=0.1,
                use_causal_mask=use_causal_mask,
            )
            # Coarse dynamics classification head: global_dyn_token (1152) → 2 classes
            self.coarse_dyn_head = nn.Sequential(
                nn.LayerNorm(head_dim),
                nn.Linear(head_dim, head_dim // 4),
                nn.GELU(),
                nn.Linear(head_dim // 4, 2),
            )

    def forward(self, eeg):
        """
        Args:
            eeg: (B, 5, 64, 800)
        Returns:
            dict with keys:
                "eeg_cls": (B, 1152) CLS token for CLIP alignment
                "eeg_spatial": (B, 226, 2048) raw spatial features
                "fast_feat": (B, 226, 2048) features for gated fusion
                "eeg_cls_proj": (B, 1152) projected CLS for distillation
                "eeg_pooled_proj": (B, 2048) projected pooled features for distillation
            When use_temporal_dynamics=True, additionally:
                "temporal_tokens": (B, T, 1152) per-frame temporal features
                "global_dyn_token": (B, 1152) global dynamics summary
                "flow_traj_pred": (B, T) predicted per-frame flow magnitude
                "dyn_logits": (B, 2) coarse dynamics classification logits
        """
        eeg_cls, eeg_spatial = self.eeg_encoder(eeg)

        # Temporal-aware pooled features
        pooled_feat = self.temporal_pool(eeg_spatial)  # (B, 2048)

        # Distillation projections
        eeg_cls_proj = self.cls_projector(eeg_cls)      # (B, 1152)
        eeg_pooled_proj = self.spatial_projector(pooled_feat)  # (B, 2048)

        out = {
            "eeg_cls": eeg_cls,
            "eeg_spatial": eeg_spatial,
            "fast_feat": eeg_spatial,
            "eeg_cls_proj": eeg_cls_proj,
            "eeg_pooled_proj": eeg_pooled_proj,
        }

        # P1: Temporal Dynamics Decoder
        if self.use_temporal_dynamics:
            temporal_tokens, global_dyn_token, flow_traj_pred = self.temporal_decoder(eeg_spatial)
            dyn_logits = self.coarse_dyn_head(global_dyn_token)
            out["temporal_tokens"] = temporal_tokens
            out["global_dyn_token"] = global_dyn_token
            out["flow_traj_pred"] = flow_traj_pred
            out["dyn_logits"] = dyn_logits

        return out
