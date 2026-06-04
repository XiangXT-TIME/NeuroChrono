import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from local_config import get_paths
_paths = get_paths()
data_root = _paths["dataset_root"]
video_dir = os.path.basename(_paths["video_dir"])

with open(os.path.join(data_root, "captions-qwen-2.5-vl-7b.json")) as f:
    captions_raw = json.load(f)

captions = {}
for c in captions_raw:
    vid_id = int(os.path.basename(c["video"]).replace(".mp4", ""))
    captions[vid_id] = c["text"]
print(f"Caption dict: {len(captions)} entries")

# Sub 1/2/6: ep 1-20, test = ep 19-20 = videos 4860-5399
# Sub 3/4/5: ep 1-10 + ep 21-30, test = ep 29-30 = videos 7560-8099
subject_config = {
    "0001": "A", "0002": "A", "0006": "A",
    "0003": "B", "0004": "B", "0005": "B",
}

def generate_test_json(sub_id, group):
    entries = []
    for clip_idx in range(4860, 5400):
        if group == "A":
            video_id = clip_idx
        else:
            video_id = 5400 + (clip_idx - 2700)

        video_path = os.path.join(data_root, video_dir, f"{str(video_id).zfill(6)}.mp4")
        fmri_base = clip_idx * 5
        fmri_paths = [os.path.join(data_root, f"sub-{sub_id}", "visual_audio", f"{fmri_base+j}.npy") for j in range(5)]
        eeg_paths = [os.path.join(data_root, f"sub-{sub_id}", "eeg_02", f"{fmri_base+j}.npy") for j in range(5)]
        text = captions.get(video_id, "")

        entries.append({"video": video_path, "fmri": fmri_paths, "eeg": eeg_paths, "text": text})
    return entries

for sub_id, group in subject_config.items():
    if sub_id == "0005":
        # Verify existing
        with open(os.path.join(data_root, "sub-0005_test_va.json")) as f:
            d5 = json.load(f)
        gen = generate_test_json("0005", "B")
        ok = d5[0]["video"] == gen[0]["video"] and d5[-1]["video"] == gen[-1]["video"]
        print(f"sub-0005: existing {len(d5)} samples, verification {'MATCH' if ok else 'MISMATCH'}")
        continue

    entries = generate_test_json(sub_id, group)
    out_path = os.path.join(data_root, f"sub-{sub_id}_test_va.json")
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)
    v0 = os.path.basename(entries[0]["video"]).replace(".mp4", "")
    v1 = os.path.basename(entries[-1]["video"]).replace(".mp4", "")
    print(f"sub-{sub_id} (group {group}): {len(entries)} samples, videos {v0}-{v1}")
