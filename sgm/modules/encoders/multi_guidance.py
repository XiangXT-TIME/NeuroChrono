"""Multi-Guidance Decoder Adapter v2: combine guidance channels into DiT context.

v2 changes: motion guidance now uses distilled EEG pooled features (2048-dim)
instead of classification head outputs (140-dim cat of dyn/mot/tc/dir).

P1 extension: optional gated residual adapter for temporal dynamics guidance.
"""
import torch
import torch.nn as nn


class MultiGuidanceAdapter(nn.Module):
    """
    Computes guidance signals and combines them with fused brain latent
    to produce the final context tensor for the DiT decoder.

    v2: motion guidance uses eeg_pooled_proj (2048-dim distilled features)
    instead of concatenated classification outputs (140-dim).

    P1: optional temporal guidance via gated residual from global_dyn_token.
    """
    def __init__(
        self,
        brain_dim=4096,
        head_dim=1152,
        num_spatial=226,
        use_keyframe_guidance=True,
        use_text_guidance=True,
        use_motion_guidance=True,
        use_brain_latent_guidance=True,
        mot_input_dim=2048,  # v2: distilled EEG feature dim (was 140)
        # P1: temporal dynamics guidance
        use_temporal_guidance=False,
    ):
        super().__init__()
        self.use_keyframe_guidance = use_keyframe_guidance
        self.use_text_guidance = use_text_guidance
        self.use_motion_guidance = use_motion_guidance
        self.use_brain_latent_guidance = use_brain_latent_guidance
        self.use_temporal_guidance = use_temporal_guidance
        self.mot_input_dim = mot_input_dim

        if use_keyframe_guidance:
            self.key_proj = nn.Linear(head_dim, brain_dim)
        if use_text_guidance:
            self.txt_proj = nn.Linear(head_dim, brain_dim)
        if use_motion_guidance:
            self.mot_proj = nn.Linear(mot_input_dim, brain_dim)

        # P1: temporal dynamics gated residual adapter
        if use_temporal_guidance:
            self.temporal_proj = nn.Linear(head_dim, brain_dim)
            self.temporal_gate = nn.Linear(head_dim, 1)

        self.out_proj = nn.Sequential(
            nn.LayerNorm(brain_dim),
            nn.Linear(brain_dim, brain_dim),
        )

    def compute_components(self, slow_out, fast_out):
        """Project raw branch outputs into brain_dim guidance vectors.

        Returns a dict with keys in {"g_key", "g_txt", "g_mot"} when the
        corresponding channel is enabled; each tensor is (B, brain_dim).
        Temporal residual (P1) is baked into g_mot so downstream mixing stays
        a pure linear combination of 4 alpha weights.
        """
        components = {}
        if self.use_keyframe_guidance and "z_key" in slow_out:
            components["g_key"] = self.key_proj(slow_out["z_key"])
        if self.use_text_guidance and "z_txt" in slow_out:
            components["g_txt"] = self.txt_proj(slow_out["z_txt"])
        if self.use_motion_guidance:
            eeg_feat = fast_out.get("eeg_pooled_proj", None)
            if eeg_feat is not None:
                g_mot = self.mot_proj(eeg_feat)  # (B, brain_dim)
                if self.use_temporal_guidance and "global_dyn_token" in fast_out:
                    g_temporal = self.temporal_proj(fast_out["global_dyn_token"])
                    gate = torch.sigmoid(self.temporal_gate(fast_out["global_dyn_token"]))
                    g_mot = g_mot + gate * g_temporal
                components["g_mot"] = g_mot
        return components

    def mix_context(self, z_b, alphas, components):
        """Blend precomputed components with per-step alphas into a context.

        Applies the same formula as `forward`, but operates on the output of
        `compute_components` so callers (e.g. the sampler) can reuse cached
        components across timesteps with different alphas.
        """
        context = z_b.clone()
        if "g_key" in components:
            context = context + alphas["alpha_key"].unsqueeze(-1) * components["g_key"].unsqueeze(1)
        if "g_txt" in components:
            context = context + alphas["alpha_txt"].unsqueeze(-1) * components["g_txt"].unsqueeze(1)
        if "g_mot" in components:
            context = context + alphas["alpha_mot"].unsqueeze(-1) * components["g_mot"].unsqueeze(1)
        if self.use_brain_latent_guidance:
            context = context + alphas["alpha_brain"].unsqueeze(-1) * z_b
        return self.out_proj(context)

    def forward(self, z_b, alphas, slow_out, fast_out):
        """
        Args:
            z_b: (B, S, brain_dim) fused brain latent
            alphas: dict of (B, 1) weights
            slow_out: dict from SlowBranch with z_key, z_txt
            fast_out: dict from FastBranch with eeg_pooled_proj, and optionally global_dyn_token
        Returns:
            context: (B, S, brain_dim) final conditioning for DiT
        """
        components = self.compute_components(slow_out, fast_out)
        return self.mix_context(z_b, alphas, components)
