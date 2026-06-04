import os
import math
import argparse
from typing import List, Union
from tqdm import tqdm
from omegaconf import ListConfig
from PIL import Image
import imageio

import torch
# §6 perturbation experiments: opt-in CUDA determinism for bit-reproducible
# baselines (required to measure tiny α perturbation signals above the default
# GPU-nondeterminism noise floor of ~0.17/element). Path B / other default runs
# do not set FORCE_DETERMINISM, so behavior is unchanged.
if os.environ.get("FORCE_DETERMINISM", "0") == "1":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    print("[FORCE_DETERMINISM=1] CUDA/cuDNN deterministic mode enabled.")

import numpy as np
from einops import rearrange, repeat
import torchvision.transforms as TT

from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat import mpu

from diffusion_video_brain import SATVideoDiffusionEngineBrain
from arguments import get_args

import json


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = np.repeat([value_dict["prompt"]], repeats=math.prod(N)).reshape(N).tolist()
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N)).reshape(N).tolist()
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]]).to(device).repeat(*N, 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor([value_dict["crop_coords_top"], value_dict["crop_coords_left"]]).to(device).repeat(*N, 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = torch.tensor([value_dict["aesthetic_score"]]).to(device).repeat(*N, 1)
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]]).to(device).repeat(*N, 1)
            )

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]]).to(device).repeat(*N, 1)
            )
        elif key == "fps":
            batch[key] = torch.tensor([value_dict["fps"]]).to(device).repeat(math.prod(N))
        elif key == "fps_id":
            batch[key] = torch.tensor([value_dict["fps_id"]]).to(device).repeat(math.prod(N))
        elif key == "motion_bucket_id":
            batch[key] = torch.tensor([value_dict["motion_bucket_id"]]).to(device).repeat(math.prod(N))
        elif key == "pool_image":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=math.prod(N)).to(device, dtype=torch.half)
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to("cuda"),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, fps: int = 5, args=None, key=None):
    # 
    for i, vid in enumerate(video_batch):
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            gif_frames.append(frame)
        now_save_path = save_path
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)


def torch_init_model(model, init_checkpoint):
    state_dict = torch.load(init_checkpoint, map_location='cpu')["module"]
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='')
    
    print("missing keys:{}".format(missing_keys))
    print('unexpected keys:{}'.format(unexpected_keys))
    print('error msgs:{}'.format(error_msgs))

def sampling_main(args, model_cls):
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls

    load_checkpoint(model, args)
    model.eval()
    json_data = json.load(open(args.jsonpath,"r"))[:]

    sample_func = model.sample
    sdedit_strength = getattr(args, 'sdedit_strength', 1.0)
    use_dana = getattr(args, 'use_dana', False)
    if sdedit_strength < 1.0:
        print(f"[Alpha-Guidance] sdedit_strength={sdedit_strength}, skipping {int((1-sdedit_strength)*51)} early steps")
    if use_dana:
        print("[DANA] Dynamic-Aware Noise Adding enabled")
    num_samples = [1]
    force_uc_zero_embeddings = ["fmri"]
    # T = 9
    T, C = args.sampling_num_frames, args.latent_channels
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        skipped = 0
        for data_item in tqdm(json_data):
            video_path = data_item["video"]
            save_name = os.path.basename(video_path).split(".")[0]
            save_path = os.path.join(args.output_dir, save_name + ".mp4")
            if os.path.exists(save_path):
                skipped += 1
                continue
            if skipped > 0:
                print(f"Skipped {skipped} already completed samples")
                skipped = 0
            fmri_path = data_item["fmri"]
            fmri_data_list = []
            for i in range(len(fmri_path)):
                fmri = torch.from_numpy(np.load(fmri_path[i])).unsqueeze(0)
                fmri_data_list.append(fmri)
            fmri_full = torch.cat(fmri_data_list, dim=0).unsqueeze(0).cuda()
            # Split visual (8405) and auditory (10541) ROIs — matches data_video.py
            fmri = fmri_full[:, :, :8405]
            fmri_auditory = fmri_full[:, :, 8405:]

            eeg_path = data_item["eeg"]
            eeg_data_list = []
            for i in range(len(eeg_path)):
                eeg = torch.from_numpy(np.load(eeg_path[i])[:64,:]).unsqueeze(0)
                eeg_data_list.append(eeg)
            eeg = torch.cat(eeg_data_list, dim=0).unsqueeze(0).cuda()

            image_size = args.sampling_image_size
            H, W = image_size[0], image_size[1]
            F = 8  # 8x downsampled
            image = None

            mp_size = mpu.get_model_parallel_world_size()
            global_rank = torch.distributed.get_rank() // mp_size
            src = global_rank * mp_size

            batch={
                "fmri": fmri,
                "fmri_auditory": fmri_auditory,
                "eeg": eeg,
                "num_frames": 33
            }
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    print(key, batch[key].shape)
                elif isinstance(batch[key], list):
                    print(key, [len(l) for l in batch[key]])
                else:
                    print(key, batch[key])
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )

            for k in c:
                if not k == "crossattn":
                    c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))

            # DANA: extract flow_traj_pred from brain embedder
            dana_beta = None
            if use_dana:
                brain_embedder = model.conditioner.embedders[0]
                if hasattr(brain_embedder, '_last_fast_out') and 'flow_traj_pred' in brain_embedder._last_fast_out:
                    flow_traj = brain_embedder._last_fast_out['flow_traj_pred']  # (B, T_pred)
                    # Normalize to [0, 1] range using sigmoid
                    dana_beta = torch.sigmoid(flow_traj - flow_traj.mean())  # center then sigmoid

            for index in range(args.batch_size):

                samples_z = sample_func(
                    c,
                    uc=uc,
                    batch_size=1,
                    shape=(T, C, H // F, W // F),
                    sdedit_strength=sdedit_strength,
                    dana_beta=dana_beta,
                ).to("cuda")

                samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()
                # §6 perturbation experiments: save pre-VAE latent for bit-identical
                # comparisons. No-op when --save_latents is not set (Path B etc.).
                if getattr(args, 'save_latents', False) and mpu.get_model_parallel_rank() == 0:
                    os.makedirs(args.output_dir, exist_ok=True)
                    latent_path = os.path.join(args.output_dir, save_name + "_latent.pt")
                    torch.save(samples_z.detach().cpu(), latent_path)
                samples_x = model.decode_first_stage(samples_z).to(torch.float32)
                samples_x = samples_x.permute(0, 2, 1, 3, 4).contiguous()
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()
                save_path = os.path.join(
                    args.output_dir, save_name+".mp4"
                )
                # import pdb;pdb.set_trace()
                os.makedirs(args.output_dir, exist_ok=True)
                if mpu.get_model_parallel_rank() == 0:
                    save_video_as_grid_and_mp4(samples, save_path, fps=args.sampling_fps)

if __name__ == "__main__":
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--output_dir', type=str, required=True)
    py_parser.add_argument('--sdedit_strength', type=float, default=1.0,
                           help='SDEdit strength: 1.0=full denoising, 0.7=skip 30%% early steps')
    py_parser.add_argument('--use_dana', action='store_true', default=False,
                           help='Enable DANA dynamic noise (motion-aware noise mixing)')
    py_parser.add_argument('--save_latents', action='store_true', default=False,
                           help='Save pre-VAE latent (samples_z) as .pt alongside mp4 for bit-identical comparisons (§6 perturbation experiments)')
    known, args_list = py_parser.parse_known_args()

    args = get_args(args_list)
    args.output_dir = known.output_dir
    args.sdedit_strength = known.sdedit_strength
    args.use_dana = known.use_dana
    args.save_latents = known.save_latents
    del args.deepspeed_config
    args.model_config.first_stage_config.params.cp_size = 1
    args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False

    # Sampling parameters for CogVideoX-5B
    args.sampling_num_frames = 13   # VAE temporal compression: 33 video frames → 13 latent frames
    args.latent_channels = 16
    args.sampling_image_size = [480, 720]
    args.sampling_fps = 8
    args.batch_size = 1

    sampling_main(args, model_cls=SATVideoDiffusionEngineBrain)
