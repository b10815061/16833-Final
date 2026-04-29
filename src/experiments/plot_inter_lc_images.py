#!/usr/bin/env python3
"""
plot_inter_lc_images.py — Show image pairs at inter-robot loop closure moments.

For each accepted inter-robot LC, displays the two matched keyframe images
side by side so we can visually verify whether the match makes sense.

Usage:
    python3 experiments/plot_inter_lc_images.py [method] [sequence1] [sequence2]
    # defaults: dinov2  MH_01_easy  MH_02_easy
"""

import os, sys, re, glob, argparse
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

SW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_parser = argparse.ArgumentParser()
_parser.add_argument("method", nargs="?", default="dinov2")
_parser.add_argument("seq0",   nargs="?", default="MH_01_easy")
_parser.add_argument("seq1",   nargs="?", default="MH_02_easy")
_parser.add_argument("--run-dir", default=None,
                     help="Explicit run directory (skips latest-run search)")
_args = _parser.parse_args()

METHOD = _args.method
SEQ0   = _args.seq0
SEQ1   = _args.seq1

# Dataset-agnostic image dir: try TUM, then EuRoC (permanent + tmp)
def _img_dir(seq):
    tum = f"/home/ros/datasets/TUM/{seq}/rgb"
    if os.path.isdir(tum):
        return tum
    for base in ["/home/ros/datasets/EuRoC", "/tmp/euroc"]:
        euroc = f"{base}/{seq}/mav0/cam0/data"
        if os.path.isdir(euroc):
            return euroc
    return None

IMG_DIR0 = _img_dir(SEQ0)
IMG_DIR1 = _img_dir(SEQ1)

# Infer dataset prefix from where images live
def _dataset_prefix(seq):
    if os.path.isdir(f"/home/ros/datasets/TUM/{seq}"):
        return "tum"
    return "euroc"

DATASET = _dataset_prefix(SEQ0)
SEQ_KEY = f"{SEQ0}_{SEQ1}"

if _args.run_dir:
    RUN_DIR = _args.run_dir
else:
    _ma_dirs = sorted(glob.glob(os.path.join(
        SW, "results", DATASET, "multi_agent", SEQ_KEY, f"multi-agent_{METHOD}_2[0-9]*")))
    RUN_DIR = _ma_dirs[-1] if _ma_dirs else os.path.join(
        SW, "results", DATASET, "multi_agent", SEQ_KEY, f"multi-agent_{METHOD}_MISSING")
LOG_FILE = os.path.join(RUN_DIR, "ros_launch.log")
OUT_DIR  = os.path.join(RUN_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────────

def decode_kf_index(vertex_id):
    return vertex_id & 0xFFFFFFFF


def load_timestamps(csv_path):
    """Returns dict: decoded_kf_index -> timestamp_nanosec (int)."""
    mapping = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 3: continue
            vid    = int(row[0])
            sec    = int(row[1])
            nanosec = int(row[2])
            ts_ns  = sec * 10**9 + nanosec
            idx    = decode_kf_index(vid)
            mapping[idx] = ts_ns
    return mapping


def _first_ts_from_rgb_txt(seq_dir):
    rgb_txt = os.path.join(seq_dir, "rgb.txt")
    if not os.path.exists(rgb_txt):
        return None
    with open(rgb_txt) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                return float(line.split()[0])
    return None


def _parse_img_ts_ns(fname):
    """Dataset-agnostic: TUM filenames are decimal seconds, EuRoC are integer ns."""
    stem = os.path.splitext(fname)[0]
    if '.' in stem:
        return int(float(stem) * 1e9)
    return int(stem)


def find_image(img_dir, ts_ns, tol_ns=50_000_000):
    """Find image file closest to ts_ns (nanoseconds). Returns path or None."""
    best, best_diff = None, tol_ns
    for f in os.listdir(img_dir):
        if not f.endswith('.png'): continue
        try:
            diff = abs(_parse_img_ts_ns(f) - ts_ns)
            if diff < best_diff:
                best_diff = diff
                best = os.path.join(img_dir, f)
        except (ValueError, OverflowError):
            continue
    return best


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


# ── parse log for inter-robot LCs ─────────────────────────────────────────────

pattern = re.compile(
    r"New inter-robot loop closure measurement: \((\d+),(\d+)\) -> \((\d+),(\d+)\)"
)

# Deduplicate: each LC is logged twice (once per robot node), keep unique pairs
seen = set()
lc_pairs = []   # list of (robot_a, kf_a, robot_b, kf_b)

with open(LOG_FILE) as f:
    for line in f:
        m = pattern.search(line)
        if not m: continue
        ra, ka, rb, kb = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        key = (min((ra,ka),(rb,kb)), max((ra,ka),(rb,kb)))
        if key not in seen:
            seen.add(key)
            lc_pairs.append((ra, ka, rb, kb))

print(f"Found {len(lc_pairs)} unique inter-robot LCs in {METHOD} log")

# ── load timestamp mappings ────────────────────────────────────────────────────

# Find latest pose_timestamps for each robot
def find_latest_ts_csv(run_dir, robot_id):
    ros_dir = os.path.join(run_dir, "ros")
    best = None
    for root, dirs, files in os.walk(ros_dir):
        fname = f"pose_timestamps{robot_id}.csv"
        if fname in files:
            p = os.path.join(root, fname)
            if best is None or os.path.getmtime(p) > os.path.getmtime(best):
                best = p
    return best

ts_csv0 = find_latest_ts_csv(RUN_DIR, 0)
ts_csv1 = find_latest_ts_csv(RUN_DIR, 1)
print(f"Timestamps robot 0: {ts_csv0}")
print(f"Timestamps robot 1: {ts_csv1}")

ts_map0 = load_timestamps(ts_csv0)
ts_map1 = load_timestamps(ts_csv1)
print(f"  Robot 0: {len(ts_map0)} keyframes   Robot 1: {len(ts_map1)} keyframes")

# Compute per-robot timestamp offset to reverse the player's clock alignment.
# The TUM player shifts robot i's timestamps by (t0_seq0 - t0_seq_i) so all
# robots share the same clock base. We add back the offset to recover original
# dataset timestamps for image lookup.
_seq0_dir = f"/home/ros/datasets/TUM/{SEQ0}"
_seq1_dir = f"/home/ros/datasets/TUM/{SEQ1}"
_t0_seq0 = _first_ts_from_rgb_txt(_seq0_dir)
_t0_seq1 = _first_ts_from_rgb_txt(_seq1_dir)
offset_ns = {
    0: 0,
    1: int((_t0_seq1 - _t0_seq0) * 1e9) if (_t0_seq0 and _t0_seq1) else 0,
}

# ── build image pairs ──────────────────────────────────────────────────────────

pairs_found = []
for ra, ka, rb, kb in lc_pairs:
    # Normalise so robot 0 is always on the left
    if ra == 1: ra, ka, rb, kb = rb, kb, ra, ka

    ts0 = ts_map0.get(ka)
    ts1 = ts_map1.get(kb)
    if ts0 is None or ts1 is None:
        print(f"  Skip ({ra},{ka})->({rb},{kb}): timestamp not found")
        continue

    img0_path = find_image(IMG_DIR0, ts0 + offset_ns[0])
    img1_path = find_image(IMG_DIR1, ts1 + offset_ns[1])
    if img0_path is None or img1_path is None:
        print(f"  Skip ({ra},{ka})->({rb},{kb}): image not found")
        continue

    pairs_found.append((ka, kb, ts0, ts1, img0_path, img1_path))

print(f"\n{len(pairs_found)} pairs with images found")

# ── plot ──────────────────────────────────────────────────────────────────────

MAX_SHOW = min(len(pairs_found), 20)   # cap at 20 pairs
N_COLS   = 4   # 2 image columns × 2 pairs per row
N_ROWS   = int(np.ceil(MAX_SHOW / 2))

fig = plt.figure(figsize=(16, N_ROWS * 3.5))
gs  = gridspec.GridSpec(N_ROWS, N_COLS, figure=fig,
                        hspace=0.45, wspace=0.08)

for i, (ka, kb, ts0, ts1, p0, p1) in enumerate(pairs_found[:MAX_SHOW]):
    row = i // 2
    col_base = (i % 2) * 2

    img0 = load_gray(p0)
    img1 = load_gray(p1)

    ax0 = fig.add_subplot(gs[row, col_base])
    ax1 = fig.add_subplot(gs[row, col_base + 1])

    ax0.imshow(img0, cmap="gray", vmin=0, vmax=255)
    ax1.imshow(img1, cmap="gray", vmin=0, vmax=255)

    ax0.set_title(f"R0 kf={ka}\n{SEQ0}", fontsize=8, pad=2)
    ax1.set_title(f"R1 kf={kb}\n{SEQ1}", fontsize=8, pad=2)

    # Red border on left image, blue on right to distinguish robots
    for ax, color in [(ax0, "tomato"), (ax1, "steelblue")]:
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
        ax.set_xticks([]); ax.set_yticks([])

    # LC index label between the pair
    fig.text(
        (col_base + 1) / N_COLS,
        1 - (row + 0.55) / N_ROWS,
        f"LC #{i+1}",
        ha="center", va="center", fontsize=7,
        color="gray", style="italic",
        transform=fig.transFigure
    )

fig.suptitle(
    f"Inter-Robot Loop Closure Image Pairs — {METHOD.title()}\n"
    f"Red border = {SEQ0} (Robot 0)   Blue border = {SEQ1} (Robot 1)\n"
    f"Visually similar pairs = plausible LC   |   Dissimilar = false positive",
    fontsize=10, y=1.01
)

out_path = os.path.join(OUT_DIR, f"fig_inter_lc_images_{METHOD}.png")
fig.savefig(out_path, bbox_inches="tight", dpi=130)
plt.close(fig)
print(f"\n→ {out_path}")
