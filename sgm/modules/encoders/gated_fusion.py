"""Cross-Modal Gated Fusion: slow_feat + fast_feat → alpha weights + z_b"""
import torch
import torch.nn as nn
from sgm.modules.encoders.common import TransformerEncoderLayer


class CrossModalGatedFusion(nn.Module):
    """
    Learns gating weights (alpha_key, alpha_txt, alpha_mot, alpha_brain)
    and produces a fused brain latent z_b from slow and fast branch features.
    Supports fixed_weights mode for ablation.

    Direction ① Path B (t_emb_dim > 0):
      gate_net additionally consumes a timestep embedding so alpha becomes a
      learned function of (sample, tau) instead of sample only. t_emb_proj's
      last linear is zero-initialized so iter-0 behavior (t_emb contribution=0)
      matches a v2 checkpoint when gate_net[0].weight is padded on the t_emb
      columns (see tools/partial_load_v2_to_pathB.py).
    """
    def __init__(
        self,
        slow_dim=2048,
        fast_dim=2048,
        hidden_dim=2048,
        output_dim=4096,
        num_heads=16,
        num_layers=4,
        num_spatial=226,
        num_alphas=4,
        fixed_weights=False,
        dropout=0.1,
        t_emb_dim=0,
        t_emb_proj_dim=512,
        t_emb_proj_init_std=0.1,
        use_prior_schedule=False,
        prior_amp=0.5,
        prior_steepness=6.0,
        prior_midpoint=0.5,
    ):
        super().__init__()
        self.fixed_weights = fixed_weights
        self.num_alphas = num_alphas
        self.t_emb_dim = int(t_emb_dim)
        self.t_emb_proj_dim = int(t_emb_proj_dim)
        self.t_emb_proj_init_std = float(t_emb_proj_init_std)
        self.use_t_emb = self.t_emb_dim > 0
        # E4_reverse-shaped inductive prior on alpha_logit (Direction ① Path B,
        # 2026-04-19 addition). Worst-case floor: with W_t_emb=0 the gate_net's
        # t_emb path contributes nothing, so α_t only depends on prior_bias(τ)
        # and α equals `v2 α_base × E4_reverse schedule` at iter 0. Path A
        # validated this schedule produces FVD 425 (-31% vs v2 619).
        self.use_prior_schedule = bool(use_prior_schedule)
        self.prior_amp = float(prior_amp)
        self.prior_steepness = float(prior_steepness)
        self.prior_midpoint = float(prior_midpoint)
        if self.use_prior_schedule:
            # Channel order (key, txt, mot, brain) mirrors alpha_vec splits.
            # Sign: +1 for channels pushed HIGH at τ=0 (high-noise / early step),
            # -1 for channels pushed HIGH at τ=1 (low-noise / late step).
            # Matches E4_reverse: mot early strong, slow (key/txt/brain) late strong.
            self.register_buffer(
                "prior_sign", torch.tensor([-1.0, -1.0, 1.0, -1.0]), persistent=False,
            )

        # Project slow+fast concat to hidden dim
        self.input_proj = nn.Linear(slow_dim + fast_dim, hidden_dim)

        # Modality embeddings
        self.modality_embed = nn.Parameter(torch.randn(2, hidden_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_spatial * 2, hidden_dim))

        # Fusion transformer
        self.fusion_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # Output projection for z_b
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Gating network: produces alpha weights from pooled fusion features
        if not fixed_weights:
            if self.use_t_emb:
                self.t_emb_proj = nn.Sequential(
                    nn.Linear(self.t_emb_dim, self.t_emb_proj_dim),
                    nn.SiLU(),
                    nn.Linear(self.t_emb_proj_dim, self.t_emb_proj_dim),
                )
                # Warm-start bit-identicality with v2 is guaranteed by the
                # zero-pad on gate_net[0].weight's t_emb columns (partial_load);
                # t_emb_proj output is annihilated by that zero-pad no matter
                # what. Init std controls the magnitude of t_emb_feat, which
                # sets the initial gradient scale for W_t_emb (≈ t_emb_feat·σ').
                # std=0.1 (vs 0.02) gives ~5× larger initial gradient to help
                # gate_net escape the flat region faster (Option C).
                # NOT zero-init: zero would also kill ∂α/∂W2 path via
                # d_α/d_W2 ∝ W_t_emb · silu → bilinear deadlock when both are 0.
                nn.init.normal_(self.t_emb_proj[-1].weight, std=self.t_emb_proj_init_std)
                nn.init.zeros_(self.t_emb_proj[-1].bias)
                gate_in_dim = hidden_dim + self.t_emb_proj_dim
            else:
                gate_in_dim = hidden_dim

            # Note: Sigmoid is applied explicitly in forward (after optional prior_bias).
            self.gate_net = nn.Sequential(
                nn.Linear(gate_in_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, num_alphas),
            )
            # Zero-init last linear: pre-prior logit starts at 0 → sigmoid(prior_bias)
            # at iter 0 exactly encodes the prior schedule; without prior, α=0.5.
            nn.init.zeros_(self.gate_net[2].weight)
            nn.init.zeros_(self.gate_net[2].bias)
        else:
            self.register_buffer(
                "fixed_alpha", torch.ones(num_alphas) / num_alphas
            )

    def forward(self, slow_feat, fast_feat, t_emb=None, tau=None):
        """
        Args:
            slow_feat: (B, S, D_slow) from SlowBranch
            fast_feat: (B, S, D_fast) from FastBranch
            t_emb:     (B, t_emb_dim) sinusoidal timestep embedding for Path B
                       gate_net input. Ignored when use_t_emb=False.
            tau:       (B,) normalized diffusion step ∈ [0, 1] for prior_schedule
                       evaluation. τ=0 high-noise (early), τ=1 low-noise (late).
                       When None, prior_bias is not applied (initial conditioner
                       forward, before sigma is known).
        Returns:
            z_b: (B, S, output_dim) fused brain latent
            alphas: dict {"alpha_key", "alpha_txt", "alpha_mot", "alpha_brain"} each (B, 1)
        """
        B, S, _ = slow_feat.shape

        combined = torch.cat([slow_feat, fast_feat], dim=-1)  # (B, S, D_s+D_f)
        h = self.input_proj(combined)  # (B, S, hidden)

        # Two-stream with modality embeddings
        h_slow = h + self.modality_embed[0]
        h_fast = h + self.modality_embed[1]
        h = torch.cat([h_slow, h_fast], dim=1)  # (B, 2S, hidden)
        h = h + self.pos_embed[:, :h.shape[1], :]

        for layer in self.fusion_layers:
            h = layer(h)
        h = self.norm(h)

        # z_b from first S tokens (slow-aligned) projected to output_dim
        z_b = self.output_proj(h[:, :S, :])  # (B, S, output_dim)

        # Gating
        if self.fixed_weights:
            alpha_vec = self.fixed_alpha.unsqueeze(0).expand(B, -1)  # (B, 4)
        else:
            pooled = h.mean(dim=1)  # (B, hidden)
            if self.use_t_emb:
                if t_emb is None:
                    t_emb = pooled.new_zeros(B, self.t_emb_dim)
                t_emb_feat = self.t_emb_proj(t_emb.to(pooled.dtype))  # (B, t_emb_proj_dim)
                pooled = torch.cat([pooled, t_emb_feat], dim=-1)
            alpha_logit = self.gate_net(pooled)  # (B, 4) — pre-sigmoid

            # Inject E4_reverse-shaped inductive prior in logit space.
            # sched: +1 at τ=0, -1 at τ=1, sigmoid transition around midpoint.
            # prior_sign per channel: -1 for slow (key/txt/brain), +1 for mot.
            # Worst-case floor: with W_t_emb=0 (partial_load) α's τ dependency
            # comes solely from prior_bias, giving iter-0 α_t = v2 α × E4_reverse.
            if self.use_prior_schedule and tau is not None:
                tau_t = tau.to(alpha_logit.dtype).view(-1)
                sched = 1.0 - 2.0 / (
                    1.0 + torch.exp(-self.prior_steepness * (tau_t - self.prior_midpoint))
                )  # (B,)
                prior_bias = (
                    self.prior_amp
                    * sched.view(-1, 1)
                    * self.prior_sign.to(alpha_logit.dtype).view(1, -1)
                )  # (B, 4)
                alpha_logit = alpha_logit + prior_bias

            alpha_vec = torch.sigmoid(alpha_logit)  # (B, 4)

        alphas = {
            "alpha_key": alpha_vec[:, 0:1],    # (B, 1)
            "alpha_txt": alpha_vec[:, 1:2],
            "alpha_mot": alpha_vec[:, 2:3],
            "alpha_brain": alpha_vec[:, 3:4],
        }

        return z_b, alphas
