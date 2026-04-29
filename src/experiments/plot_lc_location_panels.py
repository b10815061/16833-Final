#!/usr/bin/env python3
"""
plot_lc_location_panels.py — Per-LC 4-panel figures: image + trajectory location.

For each accepted inter-robot LC, produces one PNG with:
  Row 0: Robot 0 captured image  |  Robot 0 XY trajectory with LC point marked
  Row 1: Robot 1 captured image  |  Robot 1 XY trajectory with LC point marked

Usage:
    python3 experiments/plot_lc_location_panels.py [method] [seq0] [seq1]
    # defaults: dinov2  MH_01_easy  MH_02_easy
"""

import os, sys, re, csv, pickle, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

SW     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SW, "experiments"))  # for dinov2_server classes
METHOD = sys.argv[1] if len(sys.argv) > 1 else "dinov2"
SEQ0   = sys.argv[2] if len(sys.argv) > 2 else "MH_01_easy"
SEQ1   = sys.argv[3] if len(sys.argv) > 3 else "MH_02_easy"

DATASET0 = f"/home/ros/datasets/EuRoC/{SEQ0}/mav0"
DATASET1 = f"/tmp/euroc/{SEQ1}/mav0"
IMG_DIR0 = os.path.join(DATASET0, "cam0", "data")
IMG_DIR1 = os.path.join(DATASET1, "cam0", "data")

_ma_dirs = sorted(glob.glob(os.path.join(
    SW, "results", "euroc", "multi_agent", f"multi-agent_{METHOD}_2[0-9]*")))
RUN_DIR  = _ma_dirs[-1] if _ma_dirs else os.path.join(
    SW, "results", "euroc", "multi_agent", f"multi-agent_{METHOD}_MISSING")
LOG_FILE = os.path.join(RUN_DIR, "ros_launch.log")
OUT_DIR  = os.path.join(RUN_DIR, "figures", "lc_pairs", METHOD)
os.makedirs(OUT_DIR, exist_ok=True)

COLOR_R0 = "tomato"
COLOR_R1 = "steelblue"
CACHE_DIR = "/tmp/dinov2_cache"
SIMILARITY_THRESHOLD = 0.75   # same as config

# ── helpers (reused from plot_inter_lc_images.py) ─────────────────────────────

def decode_kf_index(vertex_id):
    return vertex_id & 0xFFFFFFFF


def load_timestamps(csv_path):
    """Returns {decoded_kf_index: timestamp_ns}."""
    mapping = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 3: continue
            vid = int(row[0])
            ts_ns = int(row[1]) * 10**9 + int(row[2])
            mapping[decode_kf_index(vid)] = ts_ns
    return mapping


def find_image(img_dir, ts_ns, tol_ns=50_000_000):
    best, best_diff = None, tol_ns
    for f in os.listdir(img_dir):
        if not f.endswith('.png'): continue
        try:
            diff = abs(int(os.path.splitext(f)[0]) - ts_ns)
            if diff < best_diff:
                best_diff = diff; best = os.path.join(img_dir, f)
        except ValueError:
            continue
    return best


def find_latest_file(run_dir, pattern_fn):
    """Walk run_dir and return path of most recently modified file matching pattern_fn."""
    best = None
    for root, _, files in os.walk(run_dir):
        for f in files:
            if pattern_fn(f):
                p = os.path.join(root, f)
                if best is None or os.path.getmtime(p) > os.path.getmtime(best):
                    best = p
    return best


def load_g2o_positions(g2o_path):
    """
    Parse g2o vertices, split by upper-32 bits of vertex ID (encodes robot ID).
    The smaller upper32 = robot 0, larger = robot 1.
    Returns traj0 = {idx: (x,y)}, traj1 = {idx: (x,y)}.
    """
    groups = {}   # upper32 -> {decoded_idx: (x,y)}
    with open(g2o_path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] != "VERTEX_SE3:QUAT": continue
            vid = int(parts[1])
            upper = vid >> 32
            idx   = decode_kf_index(vid)
            x, y  = float(parts[2]), float(parts[3])
            groups.setdefault(upper, {})[idx] = (x, y)

    sorted_keys = sorted(groups.keys())
    traj0 = groups[sorted_keys[0]]
    traj1 = groups[sorted_keys[1]] if len(sorted_keys) > 1 else {}
    return traj0, traj1


# ── descriptor similarity ─────────────────────────────────────────────────────

def build_descriptor_fn(method, cache_dir=CACHE_DIR):
    """
    Returns a function img_bgr -> descriptor (np.float32, L2-normalised).
    For DINOv2: loads model + VLAD + PCA from cache.
    For baseline (CosPlace): returns None (skip similarity).
    """
    if method != "dinov2":
        return None

    from dinov2_server import DINOv2PatchExtractor, VLADAggregator
    from sklearn.preprocessing import normalize as sk_normalize

    print("Loading DINOv2 model for similarity computation …")
    extractor = DINOv2PatchExtractor(model_name="dinov2_vitb14", layer=9, device="cpu")

    # Cache stores the raw MiniBatchKMeans — wrap in VLADAggregator
    vlad_path = os.path.join(cache_dir, "vlad_k32.pkl")
    pca_path  = os.path.join(cache_dir, "pca_d128.pkl")
    with open(vlad_path, "rb") as f: kmeans = pickle.load(f)
    with open(pca_path,  "rb") as f: pca    = pickle.load(f)

    vlad = VLADAggregator(num_clusters=32, desc_dim=768)
    vlad.kmeans = kmeans
    print("  VLAD + PCA loaded from cache.")

    def describe(img_bgr):
        import torch, torchvision.transforms as T
        from PIL import Image as PILImage
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil     = PILImage.fromarray(img_rgb)
        preprocess = T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        batch = preprocess(pil).unsqueeze(0)
        with torch.no_grad():
            patches = extractor(batch)[0].numpy()   # (n_patches, d)
        desc = vlad.generate(patches)               # VLAD vector
        desc = pca.transform(desc.reshape(1, -1)).flatten().astype(np.float32)
        desc = sk_normalize(desc.reshape(1, -1)).flatten()
        return desc

    return describe


# ── parse log ─────────────────────────────────────────────────────────────────

pattern = re.compile(
    r"New inter-robot loop closure measurement: \((\d+),(\d+)\) -> \((\d+),(\d+)\)"
)
seen, lc_pairs = set(), []
with open(LOG_FILE) as f:
    for line in f:
        m = pattern.search(line)
        if not m: continue
        ra, ka, rb, kb = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        key = (min((ra,ka),(rb,kb)), max((ra,ka),(rb,kb)))
        if key not in seen:
            seen.add(key)
            lc_pairs.append((ra, ka, rb, kb))

print(f"{METHOD}: {len(lc_pairs)} unique inter-robot LCs")

# ── load data ─────────────────────────────────────────────────────────────────

ts_csv0 = find_latest_file(RUN_DIR, lambda f: f == "pose_timestamps0.csv")
ts_csv1 = find_latest_file(RUN_DIR, lambda f: f == "pose_timestamps1.csv")
ts_map0 = load_timestamps(ts_csv0)
ts_map1 = load_timestamps(ts_csv1)
print(f"  R0: {len(ts_map0)} kf   R1: {len(ts_map1)} kf")

g2o_path = find_latest_file(RUN_DIR, lambda f: f == "optimized_global_pose_graph.g2o")
traj0, traj1 = load_g2o_positions(g2o_path)
print(f"  g2o: {len(traj0)} R0 vertices, {len(traj1)} R1 vertices")

# Pre-build sorted trajectory arrays for plotting
def traj_xy(traj_dict):
    ids = sorted(traj_dict.keys())
    return np.array([traj_dict[i] for i in ids])

traj0_xy = traj_xy(traj0)
traj1_xy = traj_xy(traj1)

# Build descriptor function (loads model once, reused for all pairs)
describe = build_descriptor_fn(METHOD)

# ── generate one figure per LC pair ───────────────────────────────────────────

generated = 0
for lc_num, (ra, ka, rb, kb) in enumerate(lc_pairs, start=1):
    # Normalise: robot 0 always on top row
    if ra == 1:
        ra, ka, rb, kb = rb, kb, ra, ka

    ts0 = ts_map0.get(ka)
    ts1 = ts_map1.get(kb)
    if ts0 is None or ts1 is None:
        print(f"  LC#{lc_num}: timestamp missing — skip")
        continue

    img0_path = find_image(IMG_DIR0, ts0)
    img1_path = find_image(IMG_DIR1, ts1)
    if img0_path is None or img1_path is None:
        print(f"  LC#{lc_num}: image not found — skip")
        continue

    pos0 = traj0.get(ka)
    pos1 = traj1.get(kb)
    if pos0 is None or pos1 is None:
        print(f"  LC#{lc_num}: g2o position missing — skip")
        continue

    img0_bgr = cv2.imread(img0_path)
    img1_bgr = cv2.imread(img1_path)
    img0 = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2GRAY)
    img1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)

    # Compute descriptor similarity if available
    sim_str = ""
    if describe is not None:
        d0 = describe(img0_bgr)
        d1 = describe(img1_bgr)
        sim = float(np.dot(d0, d1))   # both L2-normalised → cosine similarity
        sim_str = f"   |   Similarity: {sim:.3f}  (threshold={SIMILARITY_THRESHOLD})"

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    fig.suptitle(
        f"Inter-Robot Loop Closure #{lc_num} — {METHOD.title()}{sim_str}\n"
        f"Robot 0 kf={ka} ({SEQ0})    Robot 1 kf={kb} ({SEQ1})",
        fontsize=10
    )

    # ── Row 0: Robot 0 ────────────────────────────────────────────────────────
    ax_img0, ax_loc0 = axes[0, 0], axes[0, 1]

    ax_img0.imshow(img0, cmap="gray", vmin=0, vmax=255)
    ax_img0.set_title(f"Robot 0 — kf={ka}", fontsize=9)
    ax_img0.set_xticks([]); ax_img0.set_yticks([])
    for spine in ax_img0.spines.values():
        spine.set_edgecolor(COLOR_R0); spine.set_linewidth(2.5)

    ax_loc0.scatter(traj0_xy[:, 0], traj0_xy[:, 1],
                    c="#cccccc", s=4, zorder=1, label="trajectory")
    ax_loc0.scatter(*pos0, c=COLOR_R0, s=180, marker="*", zorder=3,
                    edgecolors="black", linewidths=0.5, label=f"kf={ka}")
    ax_loc0.set_title(f"Robot 0 position (kf={ka})", fontsize=9)
    ax_loc0.set_xlabel("X (m)", fontsize=8); ax_loc0.set_ylabel("Y (m)", fontsize=8)
    ax_loc0.set_aspect("equal", adjustable="datalim")
    ax_loc0.tick_params(labelsize=7)
    ax_loc0.legend(fontsize=7, loc="best")

    # ── Row 1: Robot 1 ────────────────────────────────────────────────────────
    ax_img1, ax_loc1 = axes[1, 0], axes[1, 1]

    ax_img1.imshow(img1, cmap="gray", vmin=0, vmax=255)
    ax_img1.set_title(f"Robot 1 — kf={kb}", fontsize=9)
    ax_img1.set_xticks([]); ax_img1.set_yticks([])
    for spine in ax_img1.spines.values():
        spine.set_edgecolor(COLOR_R1); spine.set_linewidth(2.5)

    ax_loc1.scatter(traj1_xy[:, 0], traj1_xy[:, 1],
                    c="#cccccc", s=4, zorder=1, label="trajectory")
    ax_loc1.scatter(*pos1, c=COLOR_R1, s=180, marker="*", zorder=3,
                    edgecolors="black", linewidths=0.5, label=f"kf={kb}")
    ax_loc1.set_title(f"Robot 1 position (kf={kb})", fontsize=9)
    ax_loc1.set_xlabel("X (m)", fontsize=8); ax_loc1.set_ylabel("Y (m)", fontsize=8)
    ax_loc1.set_aspect("equal", adjustable="datalim")
    ax_loc1.tick_params(labelsize=7)
    ax_loc1.legend(fontsize=7, loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = os.path.join(OUT_DIR, f"lc_{lc_num:02d}_r0k{ka}_r1k{kb}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  LC#{lc_num}: R0 kf={ka} ↔ R1 kf={kb}  → {os.path.basename(out_path)}")
    generated += 1

print(f"\n{generated} figures → {OUT_DIR}/")
