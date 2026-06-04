"""
    Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""


import math
from typing import Dict, Union

import torch
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

from ...modules.diffusionmodules.sampling_utils import (
    get_ancestral_step,
    linear_multistep_coeff,
    to_d,
    to_neg_log_sigma,
    to_sigma,
)
from ...util import append_dims, default, instantiate_from_config

from .guiders import DynamicCFG

DEFAULT_GUIDER = {"target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"}


def timestep_embedding(tau: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal timestep embedding for a scalar tau tensor in [0, 1].

    Direction ① Path B feeds gate_net a (B, dim) embedding so alpha can be a
    learned function of (sample, tau). Mirrors DiT's timestep embedding but
    assumes tau is already normalized to roughly [0, 1] (we pass alpha_cumprod_sqrt
    directly in both training and sampling for train/infer consistency).
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=tau.device, dtype=torch.float32) / half
    )
    args = tau.float().view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


def _schedule_value(sched: Dict, tau: float, slow: bool) -> float:
    """Timestep-aware multiplicative scalar for Slow/Fast alpha channels.

    `tau` is the normalized diffusion step in [0, 1] where 0 = first sampling
    step (high noise) and 1 = last (low noise). Slow channels are boosted
    early, Fast channels boosted late.
    """
    kind = sched.get("type", "none")
    if kind == "none" or kind is None:
        return 1.0
    amp = float(sched.get("amp", 0.0))
    if amp == 0.0:
        return 1.0

    if kind == "linear":
        # slow: 1+amp at τ=0, 1-amp at τ=1; fast: mirror
        base = 1.0 - 2.0 * tau  # +1 → -1 as τ goes 0→1
    elif kind == "cosine":
        base = math.cos(math.pi * tau)  # +1 → -1 as τ goes 0→1
    elif kind == "sigmoid":
        m = float(sched.get("midpoint", 0.5))
        k = float(sched.get("steepness", 6.0))
        # 1 at τ=0 → -1 at τ=1 (approximately, with transition around m)
        base = 1.0 - 2.0 / (1.0 + math.exp(-k * (tau - m)))
    else:
        raise ValueError(f"Unknown alpha_schedule type: {kind}")

    return 1.0 + amp * (base if slow else -base)


class BaseDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        uc = default(uc, cond)

        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]]).float()

        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, uc):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc))
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d


class EDMSampler(SingleStepDiffusionSampler):
    def __init__(
        self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(
            euler_step, x, d, dt, next_sigma, denoiser, cond, uc
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
            )

        return x


class DDIMSampler(SingleStepDiffusionSampler):
    def __init__(
        self, s_noise=0.1, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, s_noise=0.0):

        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        d = to_d(x, sigma, denoised)
        dt = append_dims(next_sigma * (1 - s_noise**2)**0.5 - sigma, x.ndim)

        euler_step = x + dt * d + s_noise * append_dims(next_sigma, x.ndim) * torch.randn_like(x)

        x = self.possible_correction_step(
            euler_step, x, d, dt, next_sigma, denoiser, cond, uc
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                self.s_noise,
            )

        return x


class AncestralSampler(SingleStepDiffusionSampler):
    def __init__(self, eta=1.0, s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eta = eta
        self.s_noise = s_noise
        self.noise_sampler = lambda x: torch.randn_like(x)

    def ancestral_euler_step(self, x, denoised, sigma, sigma_down):
        d = to_d(x, sigma, denoised)
        dt = append_dims(sigma_down - sigma, x.ndim)

        return self.euler_step(x, d, dt)

    def ancestral_step(self, x, sigma, next_sigma, sigma_up):
        x = torch.where(
            append_dims(next_sigma, x.ndim) > 0.0,
            x + self.noise_sampler(x) * self.s_noise * append_dims(sigma_up, x.ndim),
            x,
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
            )

        return x


class LinearMultistepSampler(BaseDiffusionSampler):
    def __init__(
        self,
        order=4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.order = order

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        ds = []
        sigmas_cpu = sigmas.detach().cpu().numpy()
        for i in self.get_sigma_gen(num_sigmas):
            sigma = s_in * sigmas[i]
            denoised = denoiser(
                *self.guider.prepare_inputs(x, sigma, cond, uc), **kwargs
            )
            denoised = self.guider(denoised, sigma)
            d = to_d(x, sigma, denoised)
            ds.append(d)
            if len(ds) > self.order:
                ds.pop(0)
            cur_order = min(i + 1, self.order)
            coeffs = [
                linear_multistep_coeff(cur_order, sigmas_cpu, i, j)
                for j in range(cur_order)
            ]
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))

        return x


class EulerEDMSampler(EDMSampler):
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc
    ):
        return euler_step


class HeunEDMSampler(EDMSampler):
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc
    ):
        if torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            return euler_step
        else:
            denoised = self.denoise(euler_step, denoiser, next_sigma, cond, uc)
            d_new = to_d(euler_step, next_sigma, denoised)
            d_prime = (d + d_new) / 2.0

            # apply correction if noise level is not 0
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x + d_prime * dt, euler_step
            )
            return x


class EulerAncestralSampler(AncestralSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x = self.ancestral_euler_step(x, denoised, sigma, sigma_down)
        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)

        return x


class DPMPP2SAncestralSampler(AncestralSampler):
    def get_variables(self, sigma, sigma_down):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, sigma_down)]
        h = t_next - t
        s = t + 0.5 * h
        return h, s, t, t_next

    def get_mult(self, h, s, t, t_next):
        mult1 = to_sigma(s) / to_sigma(t)
        mult2 = (-0.5 * h).expm1()
        mult3 = to_sigma(t_next) / to_sigma(t)
        mult4 = (-h).expm1()

        return mult1, mult2, mult3, mult4

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, **kwargs):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x_euler = self.ancestral_euler_step(x, denoised, sigma, sigma_down)

        if torch.sum(sigma_down) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            x = x_euler
        else:
            h, s, t, t_next = self.get_variables(sigma, sigma_down)
            mult = [
                append_dims(mult, x.ndim) for mult in self.get_mult(h, s, t, t_next)
            ]

            x2 = mult[0] * x - mult[1] * denoised
            denoised2 = self.denoise(x2, denoiser, to_sigma(s), cond, uc)
            x_dpmpp2s = mult[2] * x - mult[3] * denoised2

            # apply correction if noise level is not 0
            x = torch.where(append_dims(sigma_down, x.ndim) > 0.0, x_dpmpp2s, x_euler)

        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)
        return x


class DPMPP2MSampler(BaseDiffusionSampler):
    def get_variables(self, sigma, next_sigma, previous_sigma=None):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, next_sigma)]
        h = t_next - t

        if previous_sigma is not None:
            h_last = t - to_neg_log_sigma(previous_sigma)
            r = h_last / h
            return h, r, t, t_next
        else:
            return h, None, t, t_next

    def get_mult(self, h, r, t, t_next, previous_sigma):
        mult1 = to_sigma(t_next) / to_sigma(t)
        mult2 = (-h).expm1()

        if previous_sigma is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_sigma,
        sigma,
        next_sigma,
        denoiser,
        x,
        cond,
        uc=None,
    ):
        denoised = self.denoise(x, denoiser, sigma, cond, uc)

        h, r, t, t_next = self.get_variables(sigma, next_sigma, previous_sigma)
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, t, t_next, previous_sigma)
        ]

        x_standard = mult[0] * x - mult[1] * denoised
        if old_denoised is None or torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d

            # apply correction if noise level is not 0 and not first step
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard
            )

        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * sigmas[i - 1],
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
            )

        return x

class SDEDPMPP2MSampler(BaseDiffusionSampler):
    def get_variables(self, sigma, next_sigma, previous_sigma=None):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, next_sigma)]
        h = t_next - t

        if previous_sigma is not None:
            h_last = t - to_neg_log_sigma(previous_sigma)
            r = h_last / h
            return h, r, t, t_next
        else:
            return h, None, t, t_next

    def get_mult(self, h, r, t, t_next, previous_sigma):
        mult1 = to_sigma(t_next) / to_sigma(t) * (-h).exp()
        mult2 = (-2*h).expm1()

        if previous_sigma is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_sigma,
        sigma,
        next_sigma,
        denoiser,
        x,
        cond,
        uc=None,
    ):
        denoised = self.denoise(x, denoiser, sigma, cond, uc)

        h, r, t, t_next = self.get_variables(sigma, next_sigma, previous_sigma)
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, t, t_next, previous_sigma)
        ]
        mult_noise = append_dims(next_sigma * (1 - (-2*h).exp())**0.5, x.ndim)

        x_standard = mult[0] * x - mult[1] * denoised + mult_noise * torch.randn_like(x)
        if old_denoised is None or torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d + mult_noise * torch.randn_like(x)

            # apply correction if noise level is not 0 and not first step
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard
            )

        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, scale=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * sigmas[i - 1],
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
            )

        return x

class SdeditEDMSampler(EulerEDMSampler):
    def __init__(self, edit_ratio=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.edit_ratio = edit_ratio

    def __call__(self, denoiser, image, randn, cond, uc=None, num_steps=None, edit_ratio=None):
        randn_unit = randn.clone()
        randn, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            randn, cond, uc, num_steps
        )

        if num_steps is None:
            num_steps = self.num_steps
        if edit_ratio is None:
            edit_ratio = self.edit_ratio
        x = None

        for i in self.get_sigma_gen(num_sigmas):
            if i / num_steps < edit_ratio:
                continue
            if x is None:
                x = image + randn_unit * append_dims(s_in * sigmas[i], len(randn_unit.shape))

            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
            )

        return x

class VideoDDIMSampler(BaseDiffusionSampler):

    def __init__(self, fixed_frames=0, sdedit=False, **kwargs):
        super().__init__(**kwargs)
        self.fixed_frames = fixed_frames
        self.sdedit = sdedit

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        alpha_cumprod_sqrt, timesteps = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device, return_idx=True, do_append_zero=False
        )
        alpha_cumprod_sqrt = torch.cat([alpha_cumprod_sqrt, alpha_cumprod_sqrt.new_ones([1])])
        timesteps = torch.cat([torch.tensor(list(timesteps)).new_zeros([1])-1, torch.tensor(list(timesteps))])

        uc = default(uc, cond)

        num_sigmas = len(alpha_cumprod_sqrt)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, timesteps

    def denoise(self, x, denoiser, alpha_cumprod_sqrt, cond, uc, timestep=None, idx=None, scale=None, scale_emb=None, ofs=None):
        additional_model_inputs = {}

        if ofs is not None:
            additional_model_inputs['ofs'] = ofs

        if isinstance(scale, torch.Tensor) == False and scale == 1:
            additional_model_inputs['idx'] = x.new_ones([x.shape[0]]) * timestep
            if scale_emb is not None:
                additional_model_inputs['scale_emb'] = scale_emb
            denoised = denoiser(x, alpha_cumprod_sqrt, cond, **additional_model_inputs).to(torch.float32)
        else:
            additional_model_inputs['idx'] = torch.cat([x.new_ones([x.shape[0]]) * timestep] * 2)
            denoised = denoiser(*self.guider.prepare_inputs(x, alpha_cumprod_sqrt, cond, uc), **additional_model_inputs).to(torch.float32)
            if isinstance(self.guider, DynamicCFG):
                denoised = self.guider(denoised, (1 - alpha_cumprod_sqrt**2)**0.5, step_index=self.num_steps - timestep, scale=scale)
            else:
                denoised = self.guider(denoised, (1 - alpha_cumprod_sqrt**2)**0.5, scale=scale)
        return denoised

    def sampler_step(self, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, denoiser, x, cond, uc=None, idx=None, timestep=None, scale=None, scale_emb=None, ofs=None):
        denoised = self.denoise(x, denoiser, alpha_cumprod_sqrt, cond, uc, timestep, idx, scale=scale, scale_emb=scale_emb, ofs=ofs).to(torch.float32) # 1020

        a_t = ((1-next_alpha_cumprod_sqrt**2)/(1-alpha_cumprod_sqrt**2))**0.5
        b_t = next_alpha_cumprod_sqrt - alpha_cumprod_sqrt * a_t

        x = append_dims(a_t, x.ndim) * x + append_dims(b_t, x.ndim) * denoised
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, scale=None, scale_emb=None, ofs=None): # 1020
        x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, timesteps = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * alpha_cumprod_sqrt[i],
                s_in * alpha_cumprod_sqrt[i + 1],
                denoiser,
                x,
                cond,
                uc,
                idx=self.num_steps - i,
                timestep=timesteps[-(i+1)],
                scale=scale,
                scale_emb=scale_emb,
                ofs=ofs # 1020
            )

        return x


class Image2VideoDDIMSampler(BaseDiffusionSampler):

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        alpha_cumprod_sqrt, timesteps = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device, return_idx=True
        )
        uc = default(uc, cond)

        num_sigmas = len(alpha_cumprod_sqrt)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, timesteps

    def denoise(self, x, denoiser, alpha_cumprod_sqrt, cond, uc, timestep=None):
        additional_model_inputs = {}
        additional_model_inputs['idx'] = torch.cat([x.new_ones([x.shape[0]]) * timestep] * 2)
        denoised = denoiser(*self.guider.prepare_inputs(x, alpha_cumprod_sqrt, cond, uc), **additional_model_inputs).to(
            torch.float32)
        if isinstance(self.guider, DynamicCFG):
            denoised = self.guider(denoised, (1 - alpha_cumprod_sqrt ** 2) ** 0.5, step_index=self.num_steps - timestep)
        else:
            denoised = self.guider(denoised, (1 - alpha_cumprod_sqrt ** 2) ** 0.5)
        return denoised

    def sampler_step(self, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, denoiser, x, cond, uc=None, idx=None,
                     timestep=None):
        # 此处的sigma实际上是alpha_cumprod_sqrt
        denoised = self.denoise(x, denoiser, alpha_cumprod_sqrt, cond, uc, timestep).to(torch.float32)
        if idx == 1:
            return denoised

        a_t = ((1 - next_alpha_cumprod_sqrt ** 2) / (1 - alpha_cumprod_sqrt ** 2)) ** 0.5
        b_t = next_alpha_cumprod_sqrt - alpha_cumprod_sqrt * a_t

        x = append_dims(a_t, x.ndim) * x + append_dims(b_t, x.ndim) * denoised
        return x

    def __call__(self, image, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, timesteps = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * alpha_cumprod_sqrt[i],
                s_in * alpha_cumprod_sqrt[i + 1],
                denoiser,
                x,
                cond,
                uc,
                idx=self.num_steps - i,
                timestep=timesteps[-(i + 1)]
            )

        return x

class VPSDEDPMPP2MSampler(VideoDDIMSampler):
    def __init__(self, alpha_schedule=None, path_b=None, perturb_spec=None, **kwargs):
        super().__init__(**kwargs)
        # Direction ① path A: timestep-aware multiplicative modulation of learned alphas.
        # None (or type="none") disables, reproducing v2 static behavior.
        self.alpha_schedule = alpha_schedule if alpha_schedule is None else dict(alpha_schedule)
        # Direction ① path B: per-step re-run of gate_net with a timestep embedding
        # so alpha becomes a learned function of (sample, tau).
        # path_b: None or dict{"enabled": bool, "t_emb_dim": int}. When enabled
        # and the embedder exposes _last_slow_feat/_last_fast_feat, gate_net is
        # invoked per step; otherwise this falls through harmlessly.
        self.path_b = dict(path_b) if path_b is not None else {}
        self.use_path_b = bool(self.path_b.get("enabled", False))
        self.path_b_t_emb_dim = int(self.path_b.get("t_emb_dim", 256))
        # Clamp ablation (2026-04-22): replace selected channels with pure-prior α
        # (what gate_net would output with zero learned residual). Purpose is to
        # decompose which gate_net drift channel contributes which metric (EPE vs FVD).
        # Empty list = no clamping (production). Valid entries:
        # ["alpha_key", "alpha_txt", "alpha_mot", "alpha_brain"].
        self.path_b_clamp_channels = list(self.path_b.get("clamp_channels", []))

        # Tier 3 ablation: zero out selected alpha channels at inference.
        # Valid entries: ["alpha_key", "alpha_txt", "alpha_mot", "alpha_brain"].
        self.alpha_zero_channels = list(self.path_b.get("alpha_zero_channels", []))
        # Tier 3 ablation: bypass gate_net, use pure prior α(τ) at every step.
        self.alpha_prior_only = bool(self.path_b.get("alpha_prior_only", False))

        # §6 perturbation-analysis hooks for Exp 1 (single-step α perturbation)
        # and Exp 4 (single-step latent perturbation). perturb_spec is a dict
        # with optional keys:
        #   - "alpha_brain_step"  (int | None): step i* to perturb α_brain at
        #   - "alpha_brain_delta" (float): additive offset on α_brain (default 0.0)
        #   - "latent_step"       (int | None): step i* after which to perturb x
        #   - "latent_eps"        (float): stddev of Gaussian noise on x
        #   - "latent_seed"       (int | None): seed for reproducible η direction
        # When perturb_spec is None or all fields are None/0, behavior is
        # bit-identical to the original sampler (verified by T1 unit test).
        self.perturb_spec = dict(perturb_spec) if perturb_spec is not None else {}
        # Exp 3 (interval additivity): accept int | list | {range: [a, b]} forms.
        # Normalize to a set of step indices stored in self.perturb_alpha_steps.
        _step_raw = self.perturb_spec.get("alpha_brain_step", None)
        if _step_raw is None:
            self.perturb_alpha_steps = None
        elif isinstance(_step_raw, list) or hasattr(_step_raw, "_content"):
            self.perturb_alpha_steps = set(int(s) for s in _step_raw)
        elif isinstance(_step_raw, dict) and "range" in _step_raw:
            _a, _b = _step_raw["range"]
            self.perturb_alpha_steps = set(range(int(_a), int(_b) + 1))
        else:
            self.perturb_alpha_steps = {int(_step_raw)}
        # Back-compat attribute for legacy probe scripts expecting an int:
        self.perturb_alpha_step = (
            next(iter(self.perturb_alpha_steps)) if self.perturb_alpha_steps and len(self.perturb_alpha_steps) == 1 else None
        )
        self.perturb_alpha_delta = float(self.perturb_spec.get("alpha_brain_delta", 0.0))
        self.perturb_latent_step = self.perturb_spec.get("latent_step", None)
        if self.perturb_latent_step is not None:
            self.perturb_latent_step = int(self.perturb_latent_step)
        self.perturb_latent_eps = float(self.perturb_spec.get("latent_eps", 0.0))
        self.perturb_latent_seed = self.perturb_spec.get("latent_seed", None)
        if self.perturb_latent_seed is not None:
            self.perturb_latent_seed = int(self.perturb_latent_seed)
        self.perturb_crossattn_step = self.perturb_spec.get("crossattn_step", None)
        if self.perturb_crossattn_step is not None:
            self.perturb_crossattn_step = int(self.perturb_crossattn_step)
        self.perturb_crossattn_eps = float(self.perturb_spec.get("crossattn_eps", 0.0))
        self.perturb_crossattn_seed = self.perturb_spec.get("crossattn_seed", None)
        if self.perturb_crossattn_seed is not None:
            self.perturb_crossattn_seed = int(self.perturb_crossattn_seed)

    def _apply_crossattn_perturb(self, cond, i):
        """Apply single-step Gaussian perturbation to cond[crossattn] at step i (Exp 6).

        Perturbs the full conditioning tensor regardless of which modality
        (text/brain) populates each token. Used to establish generality of
        temporal asymmetry beyond the MGA alpha parameterization.
        """
        if self.perturb_crossattn_step is None or i != self.perturb_crossattn_step:
            return cond
        if self.perturb_crossattn_eps == 0.0:
            return cond
        ca = cond.get("crossattn", None)
        if ca is None:
            return cond
        gen = None
        if self.perturb_crossattn_seed is not None:
            gen = torch.Generator(device=ca.device)
            gen.manual_seed(int(self.perturb_crossattn_seed) + int(i))
        noise = torch.randn(ca.shape, generator=gen, device=ca.device, dtype=ca.dtype)
        new_cond = dict(cond)
        new_cond["crossattn"] = ca + float(self.perturb_crossattn_eps) * noise
        return new_cond

    def _apply_alpha_perturb(self, alphas_t):
        """Apply single-step α_brain perturbation (Exp 1). Returns modified dict."""
        if self.perturb_alpha_delta == 0.0:
            return alphas_t
        alphas_out = dict(alphas_t)
        alphas_out["alpha_brain"] = alphas_out["alpha_brain"] + self.perturb_alpha_delta
        return alphas_out

    def _remix_cond_for_step(self, cond, embedder, i, num_sigmas, alpha_cumprod_sqrt=None):
        """Rebuild cond["crossattn"] for step i using per-τ alpha modulation.

        Four branches (all gated by embedder / premix existing):
          1. path_b.enabled=True → re-run gate_net(slow_feat, fast_feat, t_emb(tau))
             for a learned α(sample, τ). Requires embedder._last_slow_feat/_last_fast_feat.
          2. alpha_schedule set → Path A legacy schedule modulation (v2 inference winner).
          3. perturb_alpha_step == i → static α + Exp-1 perturbation on α_brain.
          4. Otherwise cond is returned unchanged (v2 static behavior).

        At the perturbation step, the Exp-1 offset is applied on top of whichever
        α source is active (path B's learned, path A's scheduled, or the static
        baseline). This lets us probe perturbation sensitivity regardless of
        which gating mode the underlying checkpoint uses.
        """
        if embedder is None:
            return cond
        premix = getattr(embedder, "_last_premix", None)
        if premix is None:
            return cond

        at_perturb_step = (
            self.perturb_alpha_steps is not None and i in self.perturb_alpha_steps
        )

        # ---- Path B: learned α(sample, τ) via gate_net re-run ----
        if self.use_path_b:
            slow_feat = getattr(embedder, "_last_slow_feat", None)
            fast_feat = getattr(embedder, "_last_fast_feat", None)
            gated_fusion = getattr(embedder, "gated_fusion", None)
            if slow_feat is None or fast_feat is None or gated_fusion is None:
                return cond
            # τ for t_emb: use alpha_cumprod_sqrt[i] when available (matches training
            # convention), fall back to i/(num_sigmas-2) if not wired up.
            if alpha_cumprod_sqrt is not None:
                tau = alpha_cumprod_sqrt.to(slow_feat.device)
                if tau.dim() == 0:
                    tau = tau.view(1).expand(slow_feat.shape[0])
            else:
                tau = slow_feat.new_full(
                    (slow_feat.shape[0],), i / max(num_sigmas - 2, 1),
                )
            t_emb = timestep_embedding(tau, dim=self.path_b_t_emb_dim)
            _, alphas_t = gated_fusion(slow_feat, fast_feat, t_emb=t_emb, tau=tau)
            # Clamp ablation: override selected channels with pure-prior α(τ).
            # α_prior[ch](τ) = sigmoid(prior_amp * sched(τ) * prior_sign[ch])
            # i.e. what gate_net would output with zero learned residual.
            if self.path_b_clamp_channels:
                gf = gated_fusion
                if getattr(gf, "use_prior_schedule", False):
                    sched = 1.0 - 2.0 / (
                        1.0 + torch.exp(-gf.prior_steepness * (tau - gf.prior_midpoint))
                    )
                    prior_bias_b4 = (
                        gf.prior_amp
                        * sched.view(-1, 1)
                        * gf.prior_sign.to(sched.dtype).view(1, -1)
                    )
                    alpha_prior_b4 = torch.sigmoid(prior_bias_b4)
                    ch_idx = {"alpha_key": 0, "alpha_txt": 1, "alpha_mot": 2, "alpha_brain": 3}
                    alphas_t = dict(alphas_t)
                    for ch in self.path_b_clamp_channels:
                        if ch in ch_idx:
                            col = ch_idx[ch]
                            alphas_t[ch] = alpha_prior_b4[:, col:col+1].to(alphas_t[ch].dtype)
            # Tier 3 ablation: zero out selected alpha channels
            if self.alpha_zero_channels:
                alphas_t = dict(alphas_t)
                for ch in self.alpha_zero_channels:
                    if ch in alphas_t:
                        alphas_t[ch] = torch.zeros_like(alphas_t[ch])
            # Tier 3 ablation: gate_net bypass — use pure prior α(τ)
            if self.alpha_prior_only:
                gf = gated_fusion
                if getattr(gf, "use_prior_schedule", False):
                    sched = 1.0 - 2.0 / (
                        1.0 + torch.exp(-gf.prior_steepness * (tau - gf.prior_midpoint))
                    )
                    prior_bias_b4 = (
                        gf.prior_amp
                        * sched.view(-1, 1)
                        * gf.prior_sign.to(sched.dtype).view(1, -1)
                    )
                    alpha_prior = torch.sigmoid(prior_bias_b4)
                    alphas_t = {
                        "alpha_key": alpha_prior[:, 0:1].to(slow_feat.dtype),
                        "alpha_txt": alpha_prior[:, 1:2].to(slow_feat.dtype),
                        "alpha_mot": alpha_prior[:, 2:3].to(slow_feat.dtype),
                        "alpha_brain": alpha_prior[:, 3:4].to(slow_feat.dtype),
                    }
                else:
                    alphas_t = {
                        k: torch.full((slow_feat.shape[0], 1), 0.5,
                                      device=slow_feat.device, dtype=slow_feat.dtype)
                        for k in ["alpha_key", "alpha_txt", "alpha_mot", "alpha_brain"]
                    }
            if at_perturb_step:
                alphas_t = self._apply_alpha_perturb(alphas_t)
            new_context = embedder.guidance_adapter.mix_context(
                premix["z_b"], alphas_t, premix["components"]
            )
            new_cond = dict(cond)
            new_cond["crossattn"] = new_context.to(cond["crossattn"].dtype)
            return new_cond

        # ---- Path A: hand-crafted schedule (legacy) ----
        has_schedule = (
            self.alpha_schedule is not None
            and self.alpha_schedule.get("type", "none") not in (None, "none")
        )

        # Fast path: no path_b, no schedule, no perturbation at this step → static.
        if not has_schedule and not at_perturb_step:
            return cond

        if has_schedule:
            tau = i / max(num_sigmas - 2, 1)
            alphas_base = premix["alphas_base"]
            # H** validation (2026-04-18): optional upper-bound clamp to prevent
            # modulated alpha from escaping the sigmoid training distribution [0, 1].
            # When alpha_max is set, alphas_t[k] = min(v * scale, alpha_max). This
            # tests whether 540-scale FVD collapse under positive-amp schedules is
            # driven by alpha_brain OOD (base≈0.744, pushed to 1.08 at amp=+0.5).
            alpha_max = self.alpha_schedule.get("alpha_max", None)
            alphas_t = {}
            for k, v in alphas_base.items():
                is_slow = (k != "alpha_mot")
                scale = _schedule_value(self.alpha_schedule, tau, slow=is_slow)
                modulated = v * scale
                if alpha_max is not None:
                    modulated = modulated.clamp(max=float(alpha_max))
                alphas_t[k] = modulated
        else:
            # No schedule but at perturb step: start from static α_base.
            alphas_t = {k: v.clone() for k, v in premix["alphas_base"].items()}

        if at_perturb_step:
            alphas_t = self._apply_alpha_perturb(alphas_t)

        new_context = embedder.guidance_adapter.mix_context(
            premix["z_b"], alphas_t, premix["components"]
        )
        new_cond = dict(cond)
        new_cond["crossattn"] = new_context.to(cond["crossattn"].dtype)
        return new_cond

    def _apply_latent_perturb(self, x, i):
        """Apply single-step latent perturbation (Exp 4). Called after sampler_step(i).

        When perturb_latent_step == i and perturb_latent_eps > 0, adds
        N(0, eps^2) noise to x. If perturb_latent_seed is set, uses a
        fresh Generator seeded with (seed + i) for reproducibility across
        runs with identical perturbation target.
        """
        if self.perturb_latent_step is None or i != self.perturb_latent_step:
            return x
        if self.perturb_latent_eps == 0.0:
            return x
        if self.perturb_latent_seed is not None:
            gen = torch.Generator(device=x.device)
            gen.manual_seed(int(self.perturb_latent_seed) + int(i))
            noise = torch.randn(
                x.shape, generator=gen, device=x.device, dtype=x.dtype,
            )
        else:
            noise = torch.randn_like(x)
        return x + float(self.perturb_latent_eps) * noise

    def get_variables(self, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt=None):
        alpha_cumprod = alpha_cumprod_sqrt ** 2
        lamb = ((alpha_cumprod / (1-alpha_cumprod))**0.5).log()
        next_alpha_cumprod = next_alpha_cumprod_sqrt ** 2
        lamb_next = ((next_alpha_cumprod / (1-next_alpha_cumprod))**0.5).log()
        h = lamb_next - lamb

        if previous_alpha_cumprod_sqrt is not None:
            previous_alpha_cumprod = previous_alpha_cumprod_sqrt ** 2
            lamb_previous = ((previous_alpha_cumprod / (1-previous_alpha_cumprod))**0.5).log()
            h_last = lamb - lamb_previous
            r = h_last / h
            return h, r, lamb, lamb_next
        else:
            return h, None, lamb, lamb_next

    def get_mult(self, h, r, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt):
        mult1 = ((1-next_alpha_cumprod_sqrt**2) / (1-alpha_cumprod_sqrt**2))**0.5 * (-h).exp()
        mult2 = (-2*h).expm1() * next_alpha_cumprod_sqrt

        if previous_alpha_cumprod_sqrt is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_alpha_cumprod_sqrt,
        alpha_cumprod_sqrt,
        next_alpha_cumprod_sqrt,
        denoiser,
        x,
        cond,
        uc=None,
        idx=None,
        timestep=None,
        scale=None,
        scale_emb=None,
        ofs=None # 1020
    ):
        denoised = self.denoise(x, denoiser, alpha_cumprod_sqrt, cond, uc, timestep, idx, scale=scale, scale_emb=scale_emb, ofs=ofs).to(torch.float32) # 1020
        if idx == 1:
            return denoised, denoised

        h, r, lamb, lamb_next = self.get_variables(alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt)
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt)
        ]
        mult_noise = append_dims((1-next_alpha_cumprod_sqrt**2)**0.5 * (1 - (-2*h).exp())**0.5, x.ndim)

        x_standard = mult[0] * x - mult[1] * denoised + mult_noise * torch.randn_like(x)
        if old_denoised is None or torch.sum(next_alpha_cumprod_sqrt) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d + mult_noise * torch.randn_like(x)

            x = x_advanced

        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, scale=None, scale_emb=None, ofs=None,
                 start_step=0, init_latent=None, embedder=None): # 1020 + alpha-guidance + direction-① path A
        x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, timesteps = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        # Alpha-Guidance / SDEdit: start from intermediate timestep
        if start_step > 0 and start_step < num_sigmas - 1:
            alpha_t = alpha_cumprod_sqrt[start_step]
            if init_latent is not None:
                # SDEdit mode: add noise to init_latent at intermediate level
                noise = torch.randn_like(init_latent)
                x = alpha_t * init_latent + (1 - alpha_t ** 2) ** 0.5 * noise
            else:
                # No prior: scale pure noise to intermediate noise level
                x = (1 - alpha_t ** 2) ** 0.5 * x

        if self.fixed_frames > 0:
            prefix_frames = x[:, :self.fixed_frames]
        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            # Skip steps before start_step
            if i < start_step:
                continue

            cond_t = self._remix_cond_for_step(
                cond, embedder, i, num_sigmas,
                alpha_cumprod_sqrt=alpha_cumprod_sqrt[i],
            )
            cond_t = self._apply_crossattn_perturb(cond_t, i)

            if self.fixed_frames > 0:
                if self.sdedit:
                    rd = torch.randn_like(prefix_frames)
                    noised_prefix_frames = alpha_cumprod_sqrt[i] * prefix_frames + rd * append_dims(s_in * (1 - alpha_cumprod_sqrt[i] ** 2)**0.5, len(prefix_frames.shape))
                    x = torch.cat([noised_prefix_frames, x[:, self.fixed_frames:]], dim=1)
                else:
                    x = torch.cat([prefix_frames, x[:, self.fixed_frames:]], dim=1)
            x, old_denoised = self.sampler_step(
                old_denoised if i > start_step else None,  # Reset old_denoised at start
                None if i == 0 or i == start_step else s_in * alpha_cumprod_sqrt[i - 1],
                s_in * alpha_cumprod_sqrt[i],
                s_in * alpha_cumprod_sqrt[i + 1],
                denoiser,
                x,
                cond_t,
                uc=uc,
                idx=self.num_steps - i,
                timestep=timesteps[-(i+1)],
                scale=scale,
                scale_emb=scale_emb,
                ofs=ofs # 1020
            )
            # §6 Exp 4: single-step latent perturbation applied to x_{i+1}.
            # No-op when perturb_latent_step is None or perturb_latent_eps == 0.
            x = self._apply_latent_perturb(x, i)

        if self.fixed_frames > 0:
            x = torch.cat([prefix_frames, x[:, self.fixed_frames:]], dim=1)

        return x


class VPODEDPMPP2MSampler(VideoDDIMSampler):
    def get_variables(self, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt=None):
        alpha_cumprod = alpha_cumprod_sqrt ** 2
        lamb = ((alpha_cumprod / (1-alpha_cumprod))**0.5).log()
        next_alpha_cumprod = next_alpha_cumprod_sqrt ** 2
        lamb_next = ((next_alpha_cumprod / (1-next_alpha_cumprod))**0.5).log()
        h = lamb_next - lamb

        if previous_alpha_cumprod_sqrt is not None:
            previous_alpha_cumprod = previous_alpha_cumprod_sqrt ** 2
            lamb_previous = ((previous_alpha_cumprod / (1-previous_alpha_cumprod))**0.5).log()
            h_last = lamb - lamb_previous
            r = h_last / h
            return h, r, lamb, lamb_next
        else:
            return h, None, lamb, lamb_next

    def get_mult(self, h, r, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt):
        mult1 = ((1-next_alpha_cumprod_sqrt**2) / (1-alpha_cumprod_sqrt**2))**0.5
        mult2 = (-h).expm1() * next_alpha_cumprod_sqrt

        if previous_alpha_cumprod_sqrt is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_alpha_cumprod_sqrt,
        alpha_cumprod_sqrt,
        next_alpha_cumprod_sqrt,
        denoiser,
        x,
        cond,
        uc=None,
        idx=None,
        timestep=None
    ):
        denoised = self.denoise(x, denoiser, alpha_cumprod_sqrt, cond, uc, timestep, idx).to(torch.float32)
        if idx == 1:
            return denoised, denoised

        h, r, lamb, lamb_next = self.get_variables(alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt)
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt)
        ]

        x_standard = mult[0] * x - mult[1] * denoised
        if old_denoised is None or torch.sum(next_alpha_cumprod_sqrt) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d

            x = x_advanced

        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, scale=None, **kwargs):
        x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, timesteps = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * alpha_cumprod_sqrt[i - 1],
                s_in * alpha_cumprod_sqrt[i],
                s_in * alpha_cumprod_sqrt[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
                idx=self.num_steps - i,
                timestep=timesteps[-(i+1)]
            )

        return x

class VideoDDPMSampler(VideoDDIMSampler):
    def sampler_step(self, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, denoiser, x, cond, uc=None, idx=None):
        # 此处的sigma实际上是alpha_cumprod_sqrt
        denoised = self.denoise(x, denoiser, alpha_cumprod_sqrt, cond, uc, idx*1000//self.num_steps).to(torch.float32)
        if idx == 1:
            return denoised

        alpha_sqrt = alpha_cumprod_sqrt / next_alpha_cumprod_sqrt
        x = append_dims(alpha_sqrt * (1-next_alpha_cumprod_sqrt**2) / (1-alpha_cumprod_sqrt**2), x.ndim) * x \
            + append_dims(next_alpha_cumprod_sqrt * (1-alpha_sqrt**2) / (1-alpha_cumprod_sqrt**2), x.ndim) * denoised \
            + append_dims(((1-next_alpha_cumprod_sqrt**2) * (1-alpha_sqrt**2) / (1-alpha_cumprod_sqrt**2))**0.5, x.ndim) * torch.randn_like(x)

        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * alpha_cumprod_sqrt[i],
                s_in * alpha_cumprod_sqrt[i + 1],
                denoiser,
                x,
                cond,
                uc,
                idx=self.num_steps - i
            )

        return x