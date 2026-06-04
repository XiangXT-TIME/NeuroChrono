"""
Convert CogVideoX-5B weights from HuggingFace diffusers format to SAT format.

Reverse of convert_weight_sat2hf.py.

Usage:
    python tools/convert_weight_hf2sat.py \
        --hf_model_path /data/lilehui/cinebrain/CogVideoX-5b \
        --output_path /data/lilehui/cinebrain/CogVideoX-5b-sat \
        --num_layers 42

Expects HF model directory structure:
    CogVideoX-5b/
    ├── transformer/diffusion_pytorch_model-*.safetensors
    └── vae/diffusion_pytorch_model.safetensors

Produces SAT directory structure:
    CogVideoX-5b-sat/
    ├── transformer/
    │   ├── 1000/
    │   │   └── mp_rank_00_model_states.pt
    │   └── latest
    └── vae/
        └── 3d-vae.pt
"""

import argparse
import os
import re

import torch
from safetensors import safe_open


PREFIX = "model.diffusion_model."


def load_safetensors_dir(directory):
    """Load all safetensors files from a directory into a single state dict."""
    state_dict = {}
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".safetensors"):
            path = os.path.join(directory, fname)
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
    return state_dict


# ============================================================
# Transformer: HF -> SAT
# ============================================================

def convert_transformer_hf2sat(state_dict, num_layers):
    """Convert HF diffusers transformer keys to SAT format."""
    sat_dict = {}

    # Group keys by block for special handling
    # First: collect Q/K/V and norm1/norm2 linear for merging
    qkv_groups = {}  # block_id -> {"to_q": tensor, "to_k": tensor, "to_v": tensor}
    adaln_groups = {}  # block_id -> {"norm1": tensor, "norm2": tensor}

    for key, tensor in state_dict.items():
        # --- Special keys: Q/K/V merge ---
        m = re.match(r"transformer_blocks\.(\d+)\.attn1\.(to_[qkv])\.(weight|bias)", key)
        if m:
            block_id, qkv, wb = m.groups()
            gkey = f"{block_id}.{wb}"
            if gkey not in qkv_groups:
                qkv_groups[gkey] = {}
            qkv_groups[gkey][qkv] = tensor
            continue

        # --- Special keys: adaln merge (norm1.linear + norm2.linear) ---
        m = re.match(r"transformer_blocks\.(\d+)\.(norm[12])\.linear\.(weight|bias)", key)
        if m:
            block_id, norm_id, wb = m.groups()
            gkey = f"{block_id}.{wb}"
            if gkey not in adaln_groups:
                adaln_groups[gkey] = {}
            adaln_groups[gkey][norm_id] = tensor
            continue

        # --- Special keys: norm_q / norm_k -> layernorm_list ---
        m = re.match(r"transformer_blocks\.(\d+)\.attn1\.(norm_[qk])\.(weight|bias)", key)
        if m:
            block_id, norm_type, wb = m.groups()
            if norm_type == "norm_q":
                sat_key = f"{PREFIX}transformer.query_layernorm_list.{block_id}.{wb}"
            else:
                sat_key = f"{PREFIX}transformer.key_layernorm_list.{block_id}.{wb}"
            sat_dict[sat_key] = tensor
            continue

        # --- Regular key mapping ---
        sat_key = _map_transformer_key(key)
        if sat_key is not None:
            sat_dict[sat_key] = tensor

    # Merge Q/K/V into query_key_value
    for gkey, tensors in qkv_groups.items():
        block_id, wb = gkey.split(".")
        merged = torch.cat([tensors["to_q"], tensors["to_k"], tensors["to_v"]], dim=0)
        sat_key = f"{PREFIX}transformer.layers.{block_id}.attention.query_key_value.{wb}"
        sat_dict[sat_key] = merged

    # Merge norm1.linear + norm2.linear into adaln_layer.adaLN_modulations
    # Forward: chunks[0:3]+chunks[6:9] -> norm1, chunks[3:6]+chunks[9:12] -> norm2
    # Reverse: norm1_chunks[:3] + norm2_chunks[:3] + norm1_chunks[3:] + norm2_chunks[3:] -> original 12 chunks
    for gkey, tensors in adaln_groups.items():
        block_id, wb = gkey.split(".")
        norm1 = tensors["norm1"]
        norm2 = tensors["norm2"]
        n1_chunks = torch.chunk(norm1, 6, dim=0)
        n2_chunks = torch.chunk(norm2, 6, dim=0)
        merged = torch.cat(
            list(n1_chunks[:3]) + list(n2_chunks[:3]) + list(n1_chunks[3:]) + list(n2_chunks[3:]),
            dim=0,
        )
        sat_key = f"{PREFIX}transformer.adaln_layer.adaLN_modulations.{block_id}.{wb}"
        sat_dict[sat_key] = merged

    return sat_dict


# Reverse mapping table: HF key pattern -> SAT key
# Order matters for str.replace, so we use explicit regex matching instead
_TRANSFORMER_KEY_MAP = [
    # Global keys (non-block)
    (r"^norm_final\.(.*)", PREFIX + r"transformer.final_layernorm.\1"),
    (r"^norm_out\.norm\.(.*)", PREFIX + r"mixins.final_layer.norm_final.\1"),
    (r"^norm_out\.linear\.(.*)", PREFIX + r"mixins.final_layer.adaLN_modulation.1.\1"),
    (r"^proj_out\.(.*)", PREFIX + r"mixins.final_layer.linear.\1"),
    (r"^time_embedding\.linear_1\.(.*)", PREFIX + r"time_embed.0.\1"),
    (r"^time_embedding\.linear_2\.(.*)", PREFIX + r"time_embed.2.\1"),
    (r"^ofs_embedding\.linear_1\.(.*)", PREFIX + r"ofs_embed.0.\1"),
    (r"^ofs_embedding\.linear_2\.(.*)", PREFIX + r"ofs_embed.2.\1"),
    (r"^patch_embed\.pos_embedding$", PREFIX + r"mixins.pos_embed.pos_embedding"),
    (r"^patch_embed\.(.*)", PREFIX + r"mixins.patch_embed.\1"),
    # Per-block keys
    (r"^transformer_blocks\.(\d+)\.norm1\.norm\.(.*)",
     PREFIX + r"transformer.layers.\1.input_layernorm.\2"),
    (r"^transformer_blocks\.(\d+)\.norm2\.norm\.(.*)",
     PREFIX + r"transformer.layers.\1.post_attn1_layernorm.\2"),
    (r"^transformer_blocks\.(\d+)\.attn1\.to_out\.0\.(.*)",
     PREFIX + r"transformer.layers.\1.attention.dense.\2"),
    (r"^transformer_blocks\.(\d+)\.ff\.net\.0\.proj\.(.*)",
     PREFIX + r"transformer.layers.\1.mlp.dense_h_to_4h.\2"),
    (r"^transformer_blocks\.(\d+)\.ff\.net\.2\.(.*)",
     PREFIX + r"transformer.layers.\1.mlp.dense_4h_to_h.\2"),
]


def _map_transformer_key(hf_key):
    """Map a single HF transformer key to SAT key using regex patterns."""
    for pattern, replacement in _TRANSFORMER_KEY_MAP:
        m = re.match(pattern, hf_key)
        if m:
            return re.sub(pattern, replacement, hf_key)
    # Unmatched key - skip with warning
    print(f"  [WARN] Skipping unmapped transformer key: {hf_key}")
    return None


# ============================================================
# VAE: HF -> SAT
# ============================================================

# Reverse of VAE_KEYS_RENAME_DICT from sat2hf
_VAE_KEY_MAP = [
    # Mid blocks (must come before generic resnets replacement)
    (r"^(encoder|decoder)\.mid_block\.resnets\.0\.(.*)", r"\1.mid.block_1.\2"),
    (r"^(encoder|decoder)\.mid_block\.resnets\.1\.(.*)", r"\1.mid.block_2.\2"),
    # Up blocks: reverse index (HF up_blocks.X -> SAT up.{3-X})
    (r"^decoder\.up_blocks\.(\d+)\.(.*)", None),  # handled specially
    # Down blocks
    (r"^encoder\.down_blocks\.(.*)", r"encoder.down.\1"),
    (r"^decoder\.down_blocks\.(.*)", r"decoder.down.\1"),
    # Samplers
    (r"(.*)\.downsamplers\.0\.(.*)", r"\1.downsample.\2"),
    (r"(.*)\.upsamplers\.0\.(.*)", r"\1.upsample.\2"),
    # Resnets -> block
    (r"(.*)\.resnets\.(.*)", r"\1.block.\2"),
    # Conv shortcut
    (r"(.*)\.conv_shortcut\.(.*)", r"\1.nin_shortcut.\2"),
]


def convert_vae_hf2sat(state_dict):
    """Convert HF diffusers VAE keys to SAT format."""
    sat_dict = {}

    for key, tensor in state_dict.items():
        sat_key = _map_vae_key(key)
        sat_dict[sat_key] = tensor

    return sat_dict


def _map_vae_key(hf_key):
    """Map a single HF VAE key to SAT key."""
    result = hf_key

    # Handle up_blocks index reversal first
    m = re.match(r"^decoder\.up_blocks\.(\d+)\.(.*)", result)
    if m:
        idx = int(m.group(1))
        reverse_idx = 4 - 1 - idx
        result = f"decoder.up.{reverse_idx}.{m.group(2)}"

    # Apply other mappings
    # Mid blocks
    result = re.sub(r"^(encoder|decoder)\.mid_block\.resnets\.0\.", r"\1.mid.block_1.", result)
    result = re.sub(r"^(encoder|decoder)\.mid_block\.resnets\.1\.", r"\1.mid.block_2.", result)

    # Down blocks
    result = result.replace("down_blocks.", "down.")

    # Samplers
    result = result.replace("downsamplers.0", "downsample")
    result = result.replace("upsamplers.0", "upsample")

    # Resnets -> block
    result = result.replace("resnets.", "block.")

    # Conv shortcut
    result = result.replace("conv_shortcut", "nin_shortcut")

    return result


# ============================================================
# Main
# ============================================================

def save_sat_checkpoint(state_dict, output_dir, subdir, filename="mp_rank_00_model_states.pt"):
    """Save state dict in SAT checkpoint format."""
    ckpt_dir = os.path.join(output_dir, subdir)

    if subdir == "transformer":
        iter_dir = os.path.join(ckpt_dir, "1000")
        os.makedirs(iter_dir, exist_ok=True)
        # SAT expects {"module": state_dict} wrapper
        save_dict = {"module": state_dict}
        save_path = os.path.join(iter_dir, filename)
        torch.save(save_dict, save_path)
        # Write "latest" tracker file
        with open(os.path.join(ckpt_dir, "latest"), "w") as f:
            f.write("1000")
        print(f"Saved transformer to {save_path} ({os.path.getsize(save_path) / 1e9:.2f} GB)")
    elif subdir == "vae":
        os.makedirs(ckpt_dir, exist_ok=True)
        save_path = os.path.join(ckpt_dir, "3d-vae.pt")
        torch.save(state_dict, save_path)
        print(f"Saved VAE to {save_path} ({os.path.getsize(save_path) / 1e9:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(description="Convert CogVideoX-5B from HF diffusers to SAT format")
    parser.add_argument("--hf_model_path", type=str, required=True,
                        help="Path to HF diffusers model directory (e.g., CogVideoX-5b/)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output directory for SAT format weights")
    parser.add_argument("--num_layers", type=int, default=42,
                        help="Number of transformer layers (30 for 2B, 42 for 5B)")
    parser.add_argument("--convert_transformer", action="store_true", default=True)
    parser.add_argument("--convert_vae", action="store_true", default=True)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"],
                        help="Data type for saving")
    args = parser.parse_args()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    os.makedirs(args.output_path, exist_ok=True)

    if args.convert_transformer:
        print("Loading HF transformer weights...")
        transformer_dir = os.path.join(args.hf_model_path, "transformer")
        hf_state_dict = load_safetensors_dir(transformer_dir)
        print(f"  Loaded {len(hf_state_dict)} keys")

        print("Converting transformer HF -> SAT...")
        sat_state_dict = convert_transformer_hf2sat(hf_state_dict, args.num_layers)
        print(f"  Produced {len(sat_state_dict)} SAT keys")

        # Cast dtype
        for k in sat_state_dict:
            if sat_state_dict[k].is_floating_point():
                sat_state_dict[k] = sat_state_dict[k].to(dtype)

        save_sat_checkpoint(sat_state_dict, args.output_path, "transformer")
        del hf_state_dict, sat_state_dict

    if args.convert_vae:
        print("Loading HF VAE weights...")
        vae_path = os.path.join(args.hf_model_path, "vae", "diffusion_pytorch_model.safetensors")
        with safe_open(vae_path, framework="pt", device="cpu") as f:
            hf_vae_dict = {k: f.get_tensor(k) for k in f.keys()}
        print(f"  Loaded {len(hf_vae_dict)} keys")

        print("Converting VAE HF -> SAT...")
        sat_vae_dict = convert_vae_hf2sat(hf_vae_dict)
        print(f"  Produced {len(sat_vae_dict)} SAT keys")

        save_sat_checkpoint(sat_vae_dict, args.output_path, "vae")
        del hf_vae_dict, sat_vae_dict

    print("Done!")


if __name__ == "__main__":
    main()
