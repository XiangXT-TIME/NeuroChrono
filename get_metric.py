import os, sys, time
import numpy as np
from local_config import get_paths
from models.eval_metrics import (
    load_vit_model, load_clip_model,
    clip_score_only, ssim_score_only, psnr_score_only,
    img_classify_metric, video_classify_metric,
    clip_temporal_consistency, dino_temporal_consistency,
    compute_fvd,
    hue_pcc, compute_epe, vifi_score, clip_pcc
)
import imageio.v3 as iio
import torch

def main(data_path, sub_id=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()

    # Determine video range from test JSON
    if sub_id:
        import json
        json_path = os.path.join(get_paths()["dataset_root"], f"sub-00{sub_id}_test_va.json")
        with open(json_path) as f:
            test_data = json.load(f)
        video_ids = sorted([int(os.path.basename(d["video"]).split(".")[0]) for d in test_data])
        print(f"sub-{sub_id}: {len(video_ids)} test videos, range {video_ids[0]}-{video_ids[-1]}")
    else:
        video_ids = list(range(7560, 8100))

    gt_list = []
    pred_list = []
    skipped = 0
    print('loading test results ...')
    for idx, i in enumerate(video_ids):
        if idx % 100 == 0:
            print(f'  loading {idx}/{len(video_ids)} ...')
        pred_path = os.path.join(data_path, f'{str(i).zfill(6)}.mp4')
        if not os.path.exists(pred_path):
            skipped += 1
            continue
        pred = iio.imread(pred_path, index=None)
        gt = iio.imread(
            os.path.join(get_paths()["video_dir"], f'{str(i).zfill(6)}.mp4'),
            index=None
        )
        gt_list.append(gt)
        pred_list.append(pred[:33])
    if skipped:
        print(f'  skipped {skipped} missing files')
    print(f'test results loaded in {time.time()-t0:.1f}s')

    gt_list = np.stack(gt_list)
    pred_list = np.stack(pred_list)
    print(f'pred shape: {pred_list.shape}, gt shape: {gt_list.shape}')

    n_way = [2, 50]
    num_trials = 100
    top_k = 1

    # ==================== Load models once ====================
    t1 = time.time()
    print('\nLoading shared models ...')
    vit_processor, vit_model = load_vit_model(device=device)
    clip_processor, clip_model = load_clip_model(device=device)
    print(f'  Models loaded in {time.time()-t1:.1f}s')

    # ==================== 1. Video classification ====================
    t1 = time.time()
    print('\n[1/10] Video classification (VideoMAE) ...')
    vid_acc, vid_std = video_classify_metric(
        pred_list, gt_list,
        n_way=n_way, top_k=top_k, num_trials=num_trials,
        num_frames=gt_list.shape[1], return_std=True, device=device
    )
    for i, nway in enumerate(n_way):
        print(f'  Video {nway}-way: {np.mean(vid_acc[i]):.4f} +- {np.mean(vid_std[i]):.4f}')
    print(f'  took {time.time()-t1:.1f}s')

    # ==================== 2. FVD (FP32) ====================
    t1 = time.time()
    print('\n[2/10] FVD (I3D, FP32) ...')
    fvd_score = compute_fvd(pred_list, gt_list, device=device)
    print(f'  FVD: {fvd_score:.4f}')
    print(f'  took {time.time()-t1:.1f}s')

    # ==================== 3. CTC (reuse CLIP) ====================
    t1 = time.time()
    print('\n[3/10] CTC (CLIP Temporal Consistency) ...')
    ctc_mean, ctc_std = clip_temporal_consistency(
        pred_list, device=device, preloaded=(clip_processor, clip_model)
    )
    print(f'  CTC: {ctc_mean:.4f} +- {ctc_std:.4f}')
    print(f'  took {time.time()-t1:.1f}s')

    # ==================== 4. DTC ====================
    t1 = time.time()
    print('\n[4/10] DTC (DINO Temporal Consistency) ...')
    dtc_mean, dtc_std = dino_temporal_consistency(pred_list, device=device)
    print(f'  DTC: {dtc_mean:.4f} +- {dtc_std:.4f}')
    print(f'  took {time.time()-t1:.1f}s')

    # ==================== 5. Hue-PCC ====================
    t1 = time.time()
    print('\n[5/10] Hue-PCC (Hue Pearson Correlation) ...')
    hue_mean, hue_std = hue_pcc(pred_list, gt_list)
    print(f'  Hue-PCC: {hue_mean:.4f} +- {hue_std:.4f}')
    print(f'  took {time.time()-t1:.1f}s')

    # ==================== 6. EPE (End-Point Error) ====================
    t1 = time.time()
    print('\n[6/10] EPE (Optical Flow End-Point Error) ...')
    epe_mean, epe_std = compute_epe(pred_list, gt_list)
    print(f'  EPE: {epe_mean:.4f} +- {epe_std:.4f}')
    print(f'  took {time.time()-t1:.1f}s')

    # ==================== 7. VIFI-Score (must run before CLIP-PCC) ====================
    t1 = time.time()
    print('\n[7/10] VIFI-Score (Video CLIP ViT-B/16 Similarity) ...')
    vifi_result = vifi_score(pred_list, gt_list, device=device)
    if len(vifi_result) == 3:
        vifi_mean, vifi_std, vifi_per_video = vifi_result
    else:
        vifi_mean, vifi_std = vifi_result
        vifi_per_video = None
    if not np.isnan(vifi_mean):
        print(f'  VIFI-Score: {vifi_mean:.4f} +- {vifi_std:.4f}')
    else:
        print(f'  VIFI-Score: N/A (model not available)')
    print(f'  took {time.time()-t1:.1f}s')

    # ==================== 8. CLIP-PCC (with VIFI threshold filtering) ====================
    t1 = time.time()
    print('\n[8/10] CLIP-PCC (VIFI-filtered Adjacent Frame Consistency) ...')
    clip_pcc_mean, clip_pcc_std = clip_pcc(
        pred_list, vifi_per_video=vifi_per_video, vifi_threshold=0.6,
        device=device, preloaded=(clip_processor, clip_model)
    )
    print(f'  CLIP-PCC: {clip_pcc_mean:.4f} +- {clip_pcc_std:.4f}')
    print(f'  took {time.time()-t1:.1f}s')

    # ==================== 9-11. Per-frame metrics (reuse models) ====================
    t_frame_start = time.time()
    print('\n[9-10/10] Per-frame metrics (SSIM, PSNR, CLIP, Img Classification) ...')
    n_frames = pred_list.shape[1]
    ssim_aver, ssim_std_l = [], []
    psnr_aver, psnr_std_l = [], []
    clip_aver, clip_std_l = [], []
    acc_aver = [[] for _ in range(len(n_way))]
    acc_std  = [[] for _ in range(len(n_way))]

    for fi in range(n_frames):
        t1 = time.time()
        pred_frame = pred_list[:, fi]
        gt_frame = gt_list[:, fi]

        # SSIM + PSNR (CPU)
        s_mean, s_std = ssim_score_only(pred_frame, gt_frame)
        ssim_aver.append(s_mean); ssim_std_l.append(s_std)
        p_mean, p_std = psnr_score_only(pred_frame, gt_frame)
        psnr_aver.append(p_mean); psnr_std_l.append(p_std)

        # CLIP Score (reuse model)
        c_mean, c_std = clip_score_only(pred_frame, gt_frame, device=device,
                                        preloaded=(clip_processor, clip_model))
        clip_aver.append(c_mean); clip_std_l.append(c_std)

        # Img classification (reuse model)
        a_list, s_list = img_classify_metric(
            pred_frame, gt_frame,
            n_way=n_way, top_k=top_k, num_trials=num_trials,
            return_std=True, device=device,
            preloaded=(vit_processor, vit_model)
        )
        for idx, nway in enumerate(n_way):
            acc_aver[idx].append(np.mean(a_list[idx]))
            acc_std[idx].append(np.mean(s_list[idx]))

        elapsed = time.time() - t1
        print(f'  frame {fi}/{n_frames}: ssim={s_mean:.4f} psnr={p_mean:.2f} clip={c_mean:.4f} '
              f'2way={np.mean(a_list[0]):.4f} 50way={np.mean(a_list[1]):.4f} ({elapsed:.1f}s)')

    print(f'  Per-frame total: {time.time()-t_frame_start:.1f}s')

    # ==================== Summary ====================
    print('\n' + '='*60)
    print('                  EVALUATION RESULTS')
    print('='*60)
    print(f'SSIM:           {np.mean(ssim_aver):.4f} +- {np.mean(ssim_std_l):.4f}')
    print(f'PSNR:           {np.mean(psnr_aver):.4f} +- {np.mean(psnr_std_l):.4f}')
    print(f'CLIP Score:     {np.mean(clip_aver):.4f} +- {np.mean(clip_std_l):.4f}')
    for i, nway in enumerate(n_way):
        print(f'Img {nway:2d}-way:     {np.mean(acc_aver[i]):.4f} +- {np.mean(acc_std[i]):.4f}')
    for i, nway in enumerate(n_way):
        print(f'Video {nway:2d}-way:   {np.mean(vid_acc[i]):.4f} +- {np.mean(vid_std[i]):.4f}')
    print(f'Hue-PCC:        {hue_mean:.4f} +- {hue_std:.4f}')
    if not np.isnan(vifi_mean):
        print(f'VIFI-Score:     {vifi_mean:.4f} +- {vifi_std:.4f}')
    else:
        print(f'VIFI-Score:     N/A (model not available)')
    print(f'FVD:            {fvd_score:.4f}')
    print(f'CTC:            {ctc_mean:.4f} +- {ctc_std:.4f}')
    print(f'DTC:            {dtc_mean:.4f} +- {dtc_std:.4f}')
    print(f'CLIP-PCC:       {clip_pcc_mean:.4f} +- {clip_pcc_std:.4f}')
    print(f'EPE:            {epe_mean:.4f} +- {epe_std:.4f}')
    print('='*60)
    print(f'Total time: {time.time()-t0:.1f}s')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', required=True, help='Subject ID, e.g. 01, 05')
    parser.add_argument('--data_path', default=None, help='Override result directory (for cross-subject eval)')
    args = parser.parse_args()
    data_path = args.data_path or f'results/brain_va_5b_sub{args.sub}'
    main(data_path=data_path, sub_id=args.sub)
