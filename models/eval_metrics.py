import os
import numpy as np
from typing import List, Optional, Tuple
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from transformers import AutoProcessor, CLIPVisionModelWithProjection
import torch
from einops import rearrange
import torch.nn.functional as F
from transformers import logging
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

logging.set_verbosity_error()

# Default HuggingFace cache directory (GPU nodes have no internet, use pre-downloaded models)
HF_CACHE_DIR = os.path.expanduser('~/.cache/huggingface/hub')

def _resolve_model_path(model_id):
    """Resolve HuggingFace model ID to local snapshot path if cached."""
    cache_model_dir = os.path.join(HF_CACHE_DIR, 'models--' + model_id.replace('/', '--'))
    snapshot_dir = os.path.join(cache_model_dir, 'snapshots')
    if os.path.isdir(snapshot_dir):
        snapshots = os.listdir(snapshot_dir)
        if snapshots:
            return os.path.join(snapshot_dir, snapshots[0])
    return model_id  # fallback to original ID (will try network)

# ==================== Batched feature extraction ====================

@torch.no_grad()
def _preprocess_images_gpu(images_np, target_size, mean, std):
    """Fast GPU-accelerated image preprocessing: resize + normalize."""
    from torchvision.transforms.functional import normalize as tv_normalize, resize as tv_resize
    t = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0
    t = tv_resize(t, list(target_size), antialias=True)
    t = tv_normalize(t, mean.tolist(), std.tolist())
    return t


@torch.no_grad()
def _batch_vit_logits(images_np, processor, model, device, batch_size=256):
    """Batch ViT inference with GPU-accelerated preprocessing."""
    target_size = (processor.size.get('height', 224), processor.size.get('width', 224))
    mean = torch.tensor(processor.image_mean)
    std = torch.tensor(processor.image_std)

    all_logits = []
    for i in range(0, len(images_np), batch_size):
        batch = images_np[i:i+batch_size]
        inputs = _preprocess_images_gpu(batch, target_size, mean, std)
        logits = model(inputs.to(device, torch.float16)).logits.detach().cpu()
        all_logits.append(logits)
    return torch.cat(all_logits, dim=0)


@torch.no_grad()
def _batch_clip_features(images_np, processor, model, device, batch_size=256):
    """Batch CLIP inference with GPU-accelerated preprocessing.
    processor can be AutoProcessor (has .image_processor) or raw image processor.
    """
    if hasattr(processor, 'image_processor'):
        img_proc = processor.image_processor
    else:
        img_proc = processor
    target_size = (img_proc.size.get('shortest_edge', 224),) * 2
    mean = torch.tensor(img_proc.image_mean)
    std = torch.tensor(img_proc.image_std)

    all_feats = []
    for i in range(0, len(images_np), batch_size):
        batch = images_np[i:i+batch_size]
        inputs = _preprocess_images_gpu(batch, target_size, mean, std)
        feats = model(inputs.to(device, torch.float16)).image_embeds.float().cpu()
        all_feats.append(feats)
    return torch.cat(all_feats, dim=0)


@torch.no_grad()
def _batch_videomae_logits(videos_np, processor, model, device, batch_size=16):
    """Batch VideoMAE inference with GPU-accelerated preprocessing."""
    from torchvision.transforms.functional import normalize, resize
    target_size = (processor.size.get('shortest_edge', 224),) * 2
    mean = torch.tensor(processor.image_mean, dtype=torch.float32)
    std = torch.tensor(processor.image_std, dtype=torch.float32)

    all_logits = []
    for i in range(0, len(videos_np), batch_size):
        if i % (batch_size * 10) == 0:
            print(f'      videomae progress: {i}/{len(videos_np)}')
        batch = videos_np[i:i+batch_size]
        t_batch = torch.from_numpy(batch).permute(0, 1, 4, 2, 3).float() / 255.0
        B, T, C, H, W = t_batch.shape
        t_flat = t_batch.reshape(B * T, C, H, W)
        t_resized = resize(t_flat, list(target_size), antialias=True)
        t_resized = normalize(t_resized, mean.tolist(), std.tolist())
        _, C2, h, w = t_resized.shape
        t_resized = t_resized.reshape(B, T, C2, h, w)
        logits = model(t_resized.to(device, torch.float16)).logits.detach().cpu()
        all_logits.append(logits)
    return torch.cat(all_logits, dim=0)


# ==================== Model loaders (load once, reuse) ====================

def load_vit_model(cache_dir=None, device='cuda'):
    """Load ViT model and processor once."""
    if cache_dir is None:
        cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
    vit_path = _resolve_model_path('google/vit-base-patch16-224')
    processor = ViTImageProcessor.from_pretrained(vit_path)
    model = ViTForImageClassification.from_pretrained(vit_path).to(device, torch.float16)
    model.eval()
    return processor, model


def load_clip_model(cache_dir=None, device='cuda'):
    """Load CLIP model and processor once."""
    if cache_dir is None:
        cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
    clip_path = _resolve_model_path('openai/clip-vit-base-patch32')
    processor = AutoProcessor.from_pretrained(clip_path)
    model = CLIPVisionModelWithProjection.from_pretrained(clip_path).to(device, torch.float16)
    model.eval()
    return processor, model


# ==================== N-way accuracy ====================

def n_way_top_k_acc(pred, class_id, n_way, num_trials=40, top_k=1):
    if isinstance(class_id, int):
        class_id = [class_id]
    pick_range = [i for i in np.arange(len(pred)) if i not in class_id]
    corrects = 0
    for t in range(num_trials):
        idxs_picked = np.random.choice(pick_range, n_way-1, replace=False)
        for gt_id in class_id:
            pred_picked = torch.cat([pred[gt_id].unsqueeze(0), pred[idxs_picked]])
            pred_picked = pred_picked.argsort(descending=False)[-top_k:]
            if 0 in pred_picked:
                corrects += 1
                break
    return corrects / num_trials, np.sqrt(corrects / num_trials * (1 - corrects / num_trials) / num_trials)


# ==================== Image classification (batched, model reuse) ====================

@torch.no_grad()
def img_classify_metric(
        pred_videos: np.array,
        gt_videos: np.array,
        n_way: list = [50],
        num_trials: int = 100,
        top_k: int = 1,
        cache_dir: str = '.cache',
        device: Optional[str] = 'cuda',
        return_std: bool = False,
        batch_size: int = 256,
        preloaded: Optional[Tuple] = None
):
    for nway in n_way:
        assert nway > top_k

    if preloaded is not None:
        processor, model = preloaded
    else:
        processor, model = load_vit_model(cache_dir, device)

    pred_logits = _batch_vit_logits(pred_videos, processor, model, device, batch_size)
    gt_logits = _batch_vit_logits(gt_videos, processor, model, device, batch_size)

    acc_list = [[] for _ in range(len(n_way))]
    std_list = [[] for _ in range(len(n_way))]
    for idx in range(len(pred_videos)):
        gt_class_id = gt_logits[idx].argsort(descending=False)[-3:]
        pred_out = pred_logits[idx].softmax(-1)
        for i, nway in enumerate(n_way):
            acc, std = n_way_top_k_acc(pred_out, gt_class_id, nway, num_trials, top_k)
            acc_list[i].append(acc)
            std_list[i].append(std)
    if return_std:
        return acc_list, std_list
    return acc_list


# ==================== Video classification (batched) ====================

@torch.no_grad()
def video_classify_metric(
        pred_videos: np.array,
        gt_videos: np.array,
        n_way: list = [50],
        num_trials: int = 100,
        top_k: int = 1,
        num_frames: int = 6,
        cache_dir: str = '.cache',
        device: Optional[str] = 'cuda',
        return_std: bool = False,
        batch_size: int = 16
):
    for nway in n_way:
        assert nway > top_k
    vmae_path = _resolve_model_path('MCG-NJU/videomae-base-finetuned-kinetics')
    processor = VideoMAEImageProcessor.from_pretrained(vmae_path)
    model = VideoMAEForVideoClassification.from_pretrained(vmae_path, num_frames=num_frames).to(device, torch.float16)
    model.eval()

    print(f'    [video_classify] batch computing logits for {len(pred_videos)} videos...')
    pred_logits = _batch_videomae_logits(pred_videos, processor, model, device, batch_size)
    gt_logits = _batch_videomae_logits(gt_videos, processor, model, device, batch_size)

    acc_list = [[] for _ in range(len(n_way))]
    std_list = [[] for _ in range(len(n_way))]
    for idx in range(len(pred_videos)):
        gt_class_id = gt_logits[idx].argsort(descending=False)[-3:]
        pred_out = pred_logits[idx].softmax(-1)
        for i, nway in enumerate(n_way):
            acc, std = n_way_top_k_acc(pred_out, gt_class_id, nway, num_trials, top_k)
            acc_list[i].append(acc)
            std_list[i].append(std)
    if return_std:
        return acc_list, std_list
    return acc_list


# ==================== CLIP score (batched, model reuse) ====================

@torch.no_grad()
def clip_score_only(
        pred_videos: np.array,
        gt_videos: np.array,
        cache_dir: str = '.cache',
        device: Optional[str] = 'cuda',
        batch_size: int = 256,
        preloaded: Optional[Tuple] = None
):
    """Batched CLIP score computation. Accepts preloaded=(processor, model)."""
    if preloaded is not None:
        clip_processor, clip_model = preloaded
    else:
        clip_processor, clip_model = load_clip_model(cache_dir, device)

    pred_feats = _batch_clip_features(pred_videos, clip_processor, clip_model, device, batch_size)
    gt_feats = _batch_clip_features(gt_videos, clip_processor, clip_model, device, batch_size)
    pred_feats = F.normalize(pred_feats, dim=-1)
    gt_feats = F.normalize(gt_feats, dim=-1)
    scores = F.cosine_similarity(pred_feats, gt_feats, dim=-1).numpy()
    return np.mean(scores), np.std(scores)


# ==================== Pixel-level metrics ====================

def channel_last(img):
    if img.shape[-1] == 3:
        return img
    if len(img.shape) == 3:
        img = rearrange(img, 'c h w -> h w c')
    elif len(img.shape) == 4:
        img = rearrange(img, 'f c h w -> f h w c')
    else:
        raise ValueError(f'img shape should be 3 or 4, but got {len(img.shape)}')
    return img

def _ssim_single(args):
    pred, gt = args
    return ssim(pred, gt, data_range=255, channel_axis=-1)

def _psnr_single(args):
    pred, gt = args
    return psnr(gt.astype(np.float64), pred.astype(np.float64), data_range=255)

def ssim_score_only(pred_videos: np.array, gt_videos: np.array, num_workers: int = 8, **kwargs):
    from multiprocessing import Pool
    pred_videos = channel_last(pred_videos)
    gt_videos = channel_last(gt_videos)
    with Pool(num_workers) as pool:
        scores = pool.map(_ssim_single, list(zip(pred_videos, gt_videos)))
    return np.mean(scores), np.std(scores)

def psnr_score_only(pred_videos: np.array, gt_videos: np.array, num_workers: int = 8, **kwargs):
    from multiprocessing import Pool
    pred_videos = channel_last(pred_videos)
    gt_videos = channel_last(gt_videos)
    with Pool(num_workers) as pool:
        scores = pool.map(_psnr_single, list(zip(pred_videos, gt_videos)))
    return np.mean(scores), np.std(scores)

def ssim_metric(img1, img2):
    return ssim(img1, img2, data_range=255, channel_axis=-1)


# ==================== Mind-Animator metrics ====================

def hue_pcc(pred_videos: np.array, gt_videos: np.array):
    """Hue-based Pearson Correlation Coefficient (Mind-Animator pixel-level metric).
    Computes cosine similarity between hue channels in HSV space.
    Handles both image arrays (N,H,W,C) and video arrays (N,T,H,W,C).
    """
    import cv2
    # Flatten video dims: (N,T,H,W,C) -> (N*T, H, W, C)
    if pred_videos.ndim == 5:
        n, t = pred_videos.shape[:2]
        pred_flat = pred_videos.reshape(n * t, *pred_videos.shape[2:])
        gt_flat = gt_videos.reshape(n * t, *gt_videos.shape[2:])
    else:
        pred_flat = pred_videos
        gt_flat = gt_videos
    pred_flat = channel_last(pred_flat)
    gt_flat = channel_last(gt_flat)
    scores = []
    for pred, gt in zip(pred_flat, gt_flat):
        pred_hsv = cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_RGB2HSV)
        gt_hsv = cv2.cvtColor(gt.astype(np.uint8), cv2.COLOR_RGB2HSV)
        pred_hue = pred_hsv[:, :, 0].astype(np.float64) / 180.0
        gt_hue = gt_hsv[:, :, 0].astype(np.float64) / 180.0
        norm_pred = np.sqrt(np.sum(pred_hue ** 2))
        norm_gt = np.sqrt(np.sum(gt_hue ** 2))
        if norm_pred < 1e-8 or norm_gt < 1e-8:
            scores.append(0.0)
        else:
            scores.append(np.sum(pred_hue * gt_hue) / (norm_pred * norm_gt))
    return np.mean(scores), np.std(scores)


def _compute_epe_single(args):
    """Worker function for parallel EPE computation."""
    import cv2
    pred_vid, gt_vid = args
    vid_epes = []
    for i in range(len(pred_vid) - 1):
        p0 = cv2.cvtColor(pred_vid[i].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        p1 = cv2.cvtColor(pred_vid[i + 1].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        g0 = cv2.cvtColor(gt_vid[i].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(gt_vid[i + 1].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        flow_pred = cv2.calcOpticalFlowFarneback(p0, p1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_gt = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        vid_epes.append(np.mean(np.linalg.norm(flow_gt - flow_pred, axis=-1)))
    return np.mean(vid_epes)


def compute_epe(pred_videos: np.array, gt_videos: np.array, num_workers: int = 8):
    """End-Point Error using Farneback optical flow (Mind-Animator ST-level metric).
    Uses multiprocessing for parallel computation across videos.
    """
    from multiprocessing import Pool
    args = [(pred_videos[i], gt_videos[i]) for i in range(len(pred_videos))]
    with Pool(num_workers) as pool:
        epe_scores = pool.map(_compute_epe_single, args)
    return np.mean(epe_scores), np.std(epe_scores)


@torch.no_grad()
def vifi_score(
        pred_videos: np.array,
        gt_videos: np.array,
        cache_dir: str = '.cache',
        device: Optional[str] = 'cuda',
        batch_size: int = 256
):
    """VIFI-Score: Video-level CLIP similarity using ViT-B/16 with temporal mean pooling.
    Approximates ViFi-CLIP (Rasheed et al., CVPR 2023) by averaging per-frame
    CLIP features to produce video-level representations, then computing cosine similarity.
    """
    import open_clip
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='openai', cache_dir=HF_CACHE_DIR
        )
    except RuntimeError as e:
        print(f'  [VIFI-Score] Model download failed, skipping: {e}')
        return float('nan'), float('nan')
    model = model.to(device).eval()

    n, t = pred_videos.shape[:2]

    def _extract_video_features(videos_np):
        all_frames = videos_np.reshape(n * t, *videos_np.shape[2:])
        from torchvision.transforms.functional import normalize as tv_normalize, resize as tv_resize
        frame_feats = []
        for i in range(0, len(all_frames), batch_size):
            batch = all_frames[i:i+batch_size]
            imgs = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
            imgs = tv_resize(imgs, [224, 224], antialias=True)
            imgs = tv_normalize(imgs,
                                [0.48145466, 0.4578275, 0.40821073],
                                [0.26862954, 0.26130258, 0.27577711])
            feats = model.encode_image(imgs.to(device)).float().cpu()
            frame_feats.append(feats)
        frame_feats = torch.cat(frame_feats, dim=0).view(n, t, -1)
        video_feats = frame_feats.mean(dim=1)
        return F.normalize(video_feats, dim=-1)

    pred_feats = _extract_video_features(pred_videos)
    gt_feats = _extract_video_features(gt_videos)
    per_video_scores = F.cosine_similarity(pred_feats, gt_feats, dim=-1).numpy()
    return np.mean(per_video_scores), np.std(per_video_scores), per_video_scores


@torch.no_grad()
def clip_pcc(
        pred_videos: np.array,
        vifi_per_video: np.array = None,
        vifi_threshold: float = 0.6,
        cache_dir: str = '.cache',
        device: Optional[str] = 'cuda',
        batch_size: int = 256,
        preloaded: Optional[Tuple] = None
):
    """CLIP-PCC: Adjacent frame CLIP consistency with VIFI threshold filtering.
    Per Mind-Animator: only compute for videos where VIFI > threshold, assign 0 otherwise.
    """
    if preloaded is not None:
        clip_processor, clip_model = preloaded
    else:
        clip_processor, clip_model = load_clip_model(cache_dir, device)

    n, t = pred_videos.shape[:2]
    all_frames = pred_videos.reshape(n * t, *pred_videos.shape[2:])
    all_feats = _batch_clip_features(all_frames, clip_processor, clip_model, device, batch_size)
    all_feats = F.normalize(all_feats, dim=-1).view(n, t, -1)

    scores = []
    for i in range(n):
        if vifi_per_video is not None and vifi_per_video[i] < vifi_threshold:
            scores.append(0.0)
        else:
            sims = F.cosine_similarity(all_feats[i, :-1], all_feats[i, 1:], dim=-1)
            scores.append(sims.mean().item())
    return np.mean(scores), np.std(scores)

def remove_overlap(pred_videos, gt_videos, scene_seg_list, get_scene_seg=False):
    pred_list = []
    gt_list = []
    seg_dict = {}
    for pred, gt, seg in zip(pred_videos, gt_videos, scene_seg_list):
        if '-' not in seg:
            if get_scene_seg:
                if seg not in seg_dict.keys():
                    seg_dict[seg] = seg
                    pred_list.append(pred)
                    gt_list.append(gt)
            else:
                pred_list.append(pred)
                gt_list.append(gt)
    return np.stack(pred_list), np.stack(gt_list)


# ==================== Temporal consistency (batched, model reuse) ====================

@torch.no_grad()
def clip_temporal_consistency(
        pred_videos: np.array,
        cache_dir: str = '.cache',
        device: Optional[str] = 'cuda',
        batch_size: int = 256,
        preloaded: Optional[Tuple] = None
):
    """CTC: CLIP Temporal Consistency. Accepts preloaded=(processor, model)."""
    if preloaded is not None:
        clip_processor, clip_model = preloaded
    else:
        clip_processor, clip_model = load_clip_model(cache_dir, device)

    n, t = pred_videos.shape[:2]
    all_frames = pred_videos.reshape(n * t, *pred_videos.shape[2:])
    print(f'    [CTC] batch extracting CLIP features for {len(all_frames)} frames...')
    all_feats = _batch_clip_features(all_frames, clip_processor, clip_model, device, batch_size)
    all_feats = F.normalize(all_feats, dim=-1).view(n, t, -1)

    scores = []
    for i in range(n):
        sims = F.cosine_similarity(all_feats[i, :-1], all_feats[i, 1:], dim=-1)
        scores.append(sims.mean().item())
    return np.mean(scores), np.std(scores)


@torch.no_grad()
def dino_temporal_consistency(
        pred_videos: np.array,
        cache_dir: str = '.cache',
        device: Optional[str] = 'cuda',
        batch_size: int = 256
):
    """DTC: DINO Temporal Consistency."""
    from transformers import ViTModel, ViTImageProcessor as DinoProcessor
    dino_path = _resolve_model_path('facebook/dino-vitb16')
    dino_processor = DinoProcessor.from_pretrained(dino_path)
    dino_model = ViTModel.from_pretrained(dino_path).to(device, torch.float16)
    dino_model.eval()

    n, t = pred_videos.shape[:2]
    all_frames = pred_videos.reshape(n * t, *pred_videos.shape[2:])
    print(f'    [DTC] batch extracting DINO features for {len(all_frames)} frames...')

    target_size = (dino_processor.size.get('height', 224), dino_processor.size.get('width', 224))
    mean = torch.tensor(dino_processor.image_mean)
    std = torch.tensor(dino_processor.image_std)

    all_feats = []
    for i in range(0, len(all_frames), batch_size):
        batch = all_frames[i:i+batch_size]
        inputs = _preprocess_images_gpu(batch, target_size, mean, std)
        feats = dino_model(inputs.to(device, torch.float16)).last_hidden_state[:, 0].float().cpu()
        all_feats.append(feats)
    all_feats = torch.cat(all_feats, dim=0)
    all_feats = F.normalize(all_feats, dim=-1).view(n, t, -1)

    scores = []
    for i in range(n):
        sims = F.cosine_similarity(all_feats[i, :-1], all_feats[i, 1:], dim=-1)
        scores.append(sims.mean().item())
    return np.mean(scores), np.std(scores)


# ==================== FVD ====================

def compute_fvd(
        pred_videos: np.array,
        gt_videos: np.array,
        device: Optional[str] = 'cuda',
):
    """FVD: Fréchet Video Distance using I3D features (FP32 for numerical stability)."""
    from cdfvd.fvd import cdfvd

    fvd_calculator = cdfvd(model='i3d', n_real='full', n_fake='full',
                           device=str(device), half_precision=False)
    result = fvd_calculator.compute_fvd(
        real_videos=gt_videos.astype(np.uint8),
        fake_videos=pred_videos.astype(np.uint8)
    )
    return result
