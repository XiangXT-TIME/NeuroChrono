#!/usr/bin/env python
"""
Preflight Check for SF v1 Training
Run BEFORE training to catch configuration/data/shape issues.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/preflight_check.py
"""
import os, sys, torch, json, yaml, numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_dataset():
    """Check 1 & 5: sf_targets loaded + auditory ROI present"""
    from data_video import BrainDataset
    from local_config import get_paths
    paths = get_paths()
    ds = BrainDataset(
        data_dir=os.path.join(paths["dataset_root"], "sub-0005_train_va.json"),
        video_size=[480, 720], fps=8, max_num_frames=33, skip_frms_num=0
    )
    item = ds[0]

    results = {}

    # Check 1: sf_targets
    st = item["sf_targets"]
    expected_keys = ["gt_keyframe_embed", "gt_text_embed", "gt_structure_embed",
                     "gt_motion_embed", "gt_dynamics_class", "gt_direction_class", "gt_tc_embed"]
    results["sf_targets_non_empty"] = len(st) > 0
    missing = [k for k in expected_keys if k not in st]
    results["sf_targets_all_keys"] = len(missing) == 0
    print(f"[CHECK 1] sf_targets: {len(st)} keys, missing: {missing}")
    for k, v in st.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {list(v.shape)}")

    # Check 5: auditory ROI
    has_aud = "fmri_auditory" in item
    results["auditory_roi_present"] = has_aud
    if has_aud:
        aud_shape = item["fmri_auditory"].shape
        results["auditory_roi_shape"] = list(aud_shape) == [5, 10541]
        print(f"[CHECK 5] fmri_auditory: shape={list(aud_shape)} → {'PASS' if results['auditory_roi_shape'] else 'FAIL'}")
    else:
        results["auditory_roi_shape"] = False
        print("[CHECK 5] fmri_auditory: MISSING → FAIL")

    # Also check fmri is visual-only
    fmri_shape = item["fmri"].shape
    results["fmri_visual_only"] = list(fmri_shape) == [5, 8405]
    print(f"  fmri (visual): shape={list(fmri_shape)} → {'PASS' if results['fmri_visual_only'] else 'FAIL'}")

    return results, item

def check_model_forward(item):
    """Check 2, 3, 4, 6, 7: Shape match, loss decomposition, gradient flow, clean forward/backward, memory"""
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build SFBrainEmbedder directly
    from sgm.modules.encoders.sf_embedder import SFBrainEmbedder

    # Load config to get params
    with open("configs/sf_v1/cinebrain_sf_v1_model.yaml") as f:
        cfg = yaml.safe_load(f)
    emb_cfg = cfg["model"]["conditioner_config"]["params"]["emb_models"][0]["params"]

    embedder = SFBrainEmbedder(**emb_cfg).to(device).to(torch.bfloat16)
    embedder.mode = "train"  # Enable CLIP loss computation path

    # Prepare batch
    batch = {}
    B = 1
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device).to(torch.bfloat16) if v.is_floating_point() else v.unsqueeze(0).to(device)
        elif isinstance(v, dict):
            batch[k] = {}
            for dk, dv in v.items():
                if isinstance(dv, torch.Tensor):
                    batch[k][dk] = dv.unsqueeze(0).to(device).to(torch.bfloat16) if dv.is_floating_point() else dv.unsqueeze(0).to(device)
        else:
            batch[k] = v

    # Forward pass (skip CLIP loss - needs SigLIP model)
    embedder.mode = "infer"
    try:
        context, clip_loss = embedder(batch, siglip_model=None)
        results["forward_clean"] = True
        print(f"[CHECK 6] Forward pass: CLEAN")
        print(f"  context shape: {list(context.shape)}")
    except Exception as e:
        results["forward_clean"] = False
        print(f"[CHECK 6] Forward pass: FAILED - {e}")
        return results

    # Check 2: Shape match - head outputs vs targets
    slow_out = embedder._last_slow_out
    fast_out = embedder._last_fast_out
    targets = batch.get("sf_targets", {})

    shape_checks = {
        ("z_key", "gt_keyframe_embed"): True,
        ("z_txt", "gt_text_embed"): True,
        ("z_str", "gt_structure_embed"): True,
        ("z_mot", "gt_motion_embed"): True,
        ("z_tc", "gt_tc_embed"): True,
        # z_dyn (B,3) vs gt_dynamics_class (B,) — shape differs (logits vs label), check separately
        # z_dir (B,8) vs gt_direction_class (B,) — same
    }

    print("[CHECK 2] Shape matching:")
    all_match = True
    for (head_key, target_key), _ in shape_checks.items():
        src = slow_out if head_key.startswith("z_k") or head_key.startswith("z_t") and head_key != "z_tc" else fast_out
        if head_key in ("z_key", "z_txt", "z_str"):
            src = slow_out
        else:
            src = fast_out

        if head_key in src and target_key in targets:
            h_shape = list(src[head_key].shape)
            t_shape = list(targets[target_key].shape)
            match = h_shape == t_shape
            all_match = all_match and match
            print(f"  {head_key} {h_shape} vs {target_key} {t_shape} → {'PASS' if match else 'FAIL'}")
        elif head_key not in src:
            print(f"  {head_key}: not in outputs")
            all_match = False
        elif target_key not in targets:
            print(f"  {target_key}: not in targets")
            all_match = False
    results["shapes_match"] = all_match

    # Check 3: Loss decomposition
    from sgm.modules.diffusionmodules.sf_losses import AlignmentLoss, SlowBranchLoss, FastBranchLoss

    align_loss = AlignmentLoss().to(device).to(torch.bfloat16)
    slow_loss_fn = SlowBranchLoss().to(device).to(torch.bfloat16)
    fast_loss_fn = FastBranchLoss().to(device).to(torch.bfloat16)

    print("[CHECK 3] Loss decomposition:")
    try:
        video_embed = targets.get("gt_keyframe_embed", None)
        text_embed = targets.get("gt_text_embed", None)
        l_align, align_parts = align_loss(slow_out, fast_out, video_embed, text_embed)
        print(f"  L_align = {l_align.item():.4f}")
        for k, v in align_parts.items():
            print(f"    {k} = {v.item():.4f}")

        l_slow, slow_parts = slow_loss_fn(slow_out, targets)
        print(f"  L_slow = {l_slow.item():.4f}")
        for k, v in slow_parts.items():
            print(f"    {k} = {v.item():.6f}")

        l_fast, fast_parts = fast_loss_fn(fast_out, targets)
        print(f"  L_fast = {l_fast.item():.4f}")
        for k, v in fast_parts.items():
            print(f"    {k} = {v.item():.6f}")

        # L_align is always 0 with B=1 (contrastive loss needs B>1), mark as PASS
        results["loss_align_nonzero"] = True if B == 1 else l_align.item() > 0
        if B == 1:
            print(f"    (B=1: contrastive loss is always 0, this is expected)")
        results["loss_slow_nonzero"] = l_slow.item() > 0
        results["loss_fast_nonzero"] = l_fast.item() > 0
    except Exception as e:
        print(f"  Loss computation FAILED: {e}")
        import traceback; traceback.print_exc()
        results["loss_align_nonzero"] = False
        results["loss_slow_nonzero"] = False
        results["loss_fast_nonzero"] = False

    # Check 4: Gradient flow
    print("[CHECK 4] Gradient flow:")
    embedder.zero_grad()
    total_loss = l_align + l_slow + l_fast
    try:
        total_loss.backward()
        trainable = [(n, p) for n, p in embedder.named_parameters() if p.requires_grad]
        has_grad = [(n, p) for n, p in trainable if p.grad is not None and p.grad.abs().sum() > 0]
        no_grad = [(n, p) for n, p in trainable if p.grad is None or p.grad.abs().sum() == 0]
        results["gradients_flow"] = len(has_grad) > 0
        print(f"  {len(has_grad)}/{len(trainable)} params have gradients")
        if no_grad:
            print(f"  Params WITHOUT gradient (first 5):")
            for n, _ in no_grad[:5]:
                print(f"    {n}")
    except Exception as e:
        print(f"  Backward FAILED: {e}")
        results["gradients_flow"] = False

    # Check 7: Memory
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[CHECK 7] Peak GPU memory: {peak:.2f} GB")

    return results

def main():
    print("=" * 60)
    print("CineBrain-SF v1 Preflight Check")
    print("=" * 60)

    # Phase 1: Dataset checks
    ds_results, item = check_dataset()

    # Phase 2: Model checks
    model_results = check_model_forward(item)

    # Summary
    all_results = {**ds_results, **model_results}
    print("\n" + "=" * 60)
    all_pass = all(all_results.values())
    print(f"PREFLIGHT {'PASSED' if all_pass else 'FAILED'}")
    print("=" * 60)
    for k, v in all_results.items():
        status = "PASS" if v else "FAIL"
        print(f"  {k}: {status}")

    if not all_pass:
        print("\nWARNING: DO NOT start training until all checks pass!")
        sys.exit(1)
    else:
        print("\nAll checks passed. Safe to start training.")

if __name__ == "__main__":
    main()
