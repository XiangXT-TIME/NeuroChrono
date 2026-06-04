# NeuroChrono: Temporal Dynamics-Aware Brain-to-Video Reconstruction

[![Dataset](https://img.shields.io/badge/Dataset-CineBrain-faa035.svg?logo=Huggingface)](https://huggingface.co/datasets/Fudan-fMRI/CineBrain)

## Overview

NeuroChrono reconstructs naturalistic visual perception from non-invasive neural recordings
(fMRI, EEG) into video. It builds on CogVideoX-5B with a **Slow-Fast dual-branch**
encoder and **temporal dynamics-aware** conditioning that controls *when* each neural
guidance signal takes effect during the denoising process.

Key components:
- **Slow Branch** (fMRI-driven): semantic content, spatial structure, keyframe priors
- **Fast Branch** (EEG-driven): motion dynamics, temporal changes, scene transitions
- **Cross-Modal Gated Fusion**: learned per-sample multi-modal integration
- **Multi-Guidance Adapter**: per-channel cross-attention injection into the diffusion model

This codebase is built on [CogVideoX-5B (SAT)](https://github.com/THUDM/CogVideo) with
LoRA finetuning. It supports both static timestep-aware alpha scheduling (Path A) and
learnable per-sample gating (Path B).

## Prerequisites

- GPU: NVIDIA A100/A800 80GB (1 for inference, 2–4 for training)
- Python 3.10+, PyTorch 2.6+, CUDA 12.4
- Dataset: [CineBrain](https://huggingface.co/datasets/Fudan-fMRI/CineBrain)
- Pretrained weights: CogVideoX-5B, SigLIP2

## Installation

```bash
git clone https://github.com/XiangXT-TIME/NeuroChrono.git
cd NeuroChrono
pip install -r requirements.txt
```

Copy and edit the local config:

```bash
cp local_config.example.yaml local_config.yaml
# edit local_config.yaml with your dataset and model paths
```

## Usage

### Inference

```bash
# Per-subject inference (replace sub05 with target subject)
CUDA_VISIBLE_DEVICES=0 python sample_brain_va.py \
  --base configs/sf_v1/cinebrain_sf_v3_pathB_model.yaml configs/infer_brain_va_5b_sub05.yaml \
  --seed 42 \
  --jsonpath /path/to/sub-0005_test_va.json \
  --output_dir results/output
```

### Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  train_video_fmri.py \
  --base configs/sf_v1/cinebrain_sf_v3_pathB_model.yaml configs/sf_v1/sf_v3_pathB_train.yaml \
  --seed 42
```

### Evaluation

```bash
python get_metric.py --sub 05
```

## Citation

```bibtex
@article{neurochrono,
  title={NeuroChrono: Temporal Dynamics-Aware Brain-to-Video Reconstruction},
  author={Drift},
  year={2026}
}
```

## Acknowledgements

- [CineBrain](https://github.com/yanweifu-sii/CineBrain) for the dataset and baseline codebase
- [CogVideoX](https://github.com/THUDM/CogVideo) for the video diffusion backbone
- [SAT](https://github.com/THUDM/SwissArmyTransformer) for the transformer training framework
