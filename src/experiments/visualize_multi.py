"""
visualize_multi.py — Figures for multi-agent (2-robot) Swarm-SLAM experiments.

Usage:
    python3 experiments/visualize_multi.py [seq1] [seq2]
    # seq1 defaults to MH_01_easy, seq2 defaults to MH_02_easy

Reads from:  results/euroc/multi_agent/multi-agent_{baseline,dinov2}_YYYYMMDD_HHMMSS/
Writes to:   {most recent multi-agent run}/figures/
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import csv, re, os, sys, subprocess, glob
import cv2

import argparse

SW      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_parser = argparse.ArgumentParser()
_parser.add_argument("seq1")
_parser.add_argument("seq2")
_parser.add_argument("--runs", nargs="*", default=[],
                     help="Explicit run dirs per method: 'method:path' e.g. 'baseline:/path/to/run'")
_args = _parser.parse_args()

SEQ1    = _args.seq1.split('/')[1]
SEQ2    = _args.seq2.split('/')[1]
DATASET = _args.seq1.split('/')[0]
METHODS_LIST = ["baseline", "dinov2"]

# Parse explicit run dir overrides (method:path pairs)
_run_overrides = {}
for item in (_args.runs or []):
    if ':' in item:
        _m, _p = item.split(':', 1)
        if _p:
            _run_overrides[_m] = _p

SEQ_KEY  = f"{SEQ1}_{SEQ2}"
BASE_DIR = os.path.join(SW, "results", DATASET, "multi_agent", SEQ_KEY)

# OUT_DIR: use the most recently overridden run dir, else most recent run overall
_override_dirs = [p for p in _run_overrides.values() if p]
if _override_dirs:
    OUT_DIR = os.path.join(sorted(_override_dirs)[-1], "figures")
else:
    _all_runs = sorted(glob.glob(os.path.join(BASE_DIR, "multi-agent_*_2[0-9]*")))
    OUT_DIR = os.path.join(_all_runs[-1], "figures") if _all_runs else os.path.join(BASE_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

FONT = 11
plt.rcParams.update({"font.size": FONT, "axes.titlesize": FONT+1,
                     "axes.labelsize": FONT, "figure.dpi": 150})

COLOR_GT       = "black"
COLOR_BASELINE = "#2166ac"
COLOR_DINOV2   = "#d6604d"
COLOR_R0       = "#4dac26"   # robot 0
COLOR_R1       = "#b8860b"   # robot 1


# ── helpers ────────────────────────────────────────────────────────────────────

def _latest(method, *parts):
    if method in _run_overrides:
        root = _run_overrides[method]
    else:
        dirs = sorted(glob.glob(os.path.join(BASE_DIR, f"multi-agent_{method}_2[0-9]*")))
        root = dirs[-1] if dirs else os.path.join(BASE_DIR, f"multi-agent_{method}_MISSING")
    return os.path.join(root, *parts)


def find_latest_g2o(method):
    ros_dir = _latest(method, "ros")
    result = subprocess.run(
        ["find", ros_dir, "-name", "optimized_global_pose_graph.g2o",
         "-printf", "%T@ %p\n"],
        capture_output=True, text=True)
    lines = [l for l in result.stdout.strip().splitlines() if l]
    if not lines:
        raise FileNotFoundError(f"No g2o in {ros_dir}")
    return sorted(lines)[-1].split(" ", 1)[1]


def load_tum(path):
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1:4]   # timestamp, xyz only


def align_umeyama(src, ref):
    mu_s, mu_r = src.mean(0), ref.mean(0)
    S, R = src - mu_s, ref - mu_r
    var_s = np.mean(np.sum(S**2, axis=1))
    cov = (R.T @ S) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    diag = np.diag([1, 1, np.linalg.det(U @ Vt)])
    rot   = U @ diag @ Vt
    scale = np.sum(D * np.diag(diag)) / var_s
    return (scale * (rot @ src.T)).T + (mu_r - scale * rot @ mu_s)


def load_full_gt(seq_name):
    """Load full GT xyz from original dataset file (not the eval subset).
    Returns xyz array (N,3) or None if not found."""
    seq = seq_name.split("/")[-1]   # strip tum/ or euroc/ prefix
    # TUM groundtruth.txt: ts tx ty tz qx qy qz qw (skip # comments)
    tum_path = f"/home/ros/datasets/TUM/{seq}/groundtruth.txt"
    if os.path.isfile(tum_path):
        data = np.loadtxt(tum_path, comments="#")
        return data[:, 1:4]
    # EuRoC state_groundtruth_estimate0/data.csv: ts,px,py,pz,...
    for base in ["/home/ros/datasets/EuRoC", "/tmp/euroc"]:
        euroc_path = f"{base}/{seq}/mav0/state_groundtruth_estimate0/data.csv"
        if os.path.isfile(euroc_path):
            data = np.genfromtxt(euroc_path, delimiter=",", skip_header=1)
            return data[:, 1:4]
    return None


def sync_and_align(est_path, gt_path, max_dt=0.01):
    ts_e, xyz_e = load_tum(est_path)
    ts_g, xyz_g = load_tum(gt_path)
    me, mg, mts = [], [], []
    for i, t in enumerate(ts_e):
        j = np.argmin(np.abs(ts_g - t))
        if abs(ts_g[j] - t) <= max_dt:
            me.append(xyz_e[i]); mg.append(xyz_g[j]); mts.append(t)
    me, mg = np.array(me), np.array(mg)
    return np.array(mts), align_umeyama(me, mg), mg


def compute_ape(aligned, gt):
    errs = np.linalg.norm(aligned - gt, axis=1)
    return {"max": errs.max(), "mean": errs.mean(),
            "median": np.median(errs), "rmse": np.sqrt(np.mean(errs**2))}


def parse_g2o(path):
    vertices, edges = {}, []
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p: continue
            if p[0] == "VERTEX_SE3:QUAT":
                vertices[int(p[1])] = (float(p[2]), float(p[3]), float(p[4]))
            elif p[0] == "EDGE_SE3:QUAT":
                edges.append((int(p[1]), int(p[2])))
    id_rank = {v: i for i, v in enumerate(sorted(vertices))}
    odom, lc = [], []
    for a, b in edges:
        (odom if abs(id_rank.get(a,-999) - id_rank.get(b,-999)) == 1 else lc).append((a, b))
    return vertices, odom, lc


def parse_inter_robot_lc(log_path):
    """Count intra/inter-robot loop closures and failed inter-robot attempts from log.
    Each event is logged once per robot (×2), so counts are divided by 2."""
    intra_pat  = re.compile(r"New intra-robot loop closure")
    inter_pat  = re.compile(r"New inter-robot loop closure")
    failed_pat = re.compile(r"Failed inter-robot loop closure")
    n_intra = n_inter = n_failed = 0
    with open(log_path) as f:
        for line in f:
            if intra_pat.search(line):  n_intra  += 1
            if inter_pat.search(line):  n_inter  += 1
            if failed_pat.search(line): n_failed += 1
    # Each event duplicated (logged by both robot nodes)
    return n_intra // 2, n_inter // 2, n_failed // 2


# ── LC image pair helpers ──────────────────────────────────────────────────────

def _img_dir(seq):
    """Return (path, fmt) for the image directory of a sequence.
    fmt is 'tum' or 'euroc', used to parse image filenames as timestamps."""
    tum = f"/home/ros/datasets/TUM/{seq}/rgb"
    if os.path.isdir(tum):
        return tum, "tum"
    for base in ["/home/ros/datasets/EuRoC", "/tmp/euroc"]:
        euroc = f"{base}/{seq}/mav0/cam0/data"
        if os.path.isdir(euroc):
            return euroc, "euroc"
    return None, None


def _decode_kf_index(vertex_id):
    return vertex_id & 0xFFFFFFFF


def _load_ts_map(run_dir, robot_id):
    """Returns {kf_index: timestamp_ns} from latest pose_timestampsN.csv in run_dir/ros/."""
    ros_dir = os.path.join(run_dir, "ros")
    best = None
    for root, _, files in os.walk(ros_dir):
        fname = f"pose_timestamps{robot_id}.csv"
        if fname in files:
            p = os.path.join(root, fname)
            if best is None or os.path.getmtime(p) > os.path.getmtime(best):
                best = p
    if best is None:
        return {}
    mapping = {}
    with open(best) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 3: continue
            vid = int(row[0])
            ts_ns = int(row[1]) * 10**9 + int(row[2])
            mapping[_decode_kf_index(vid)] = ts_ns
    return mapping


def _first_ts_from_rgb_txt(seq_dir):
    """Return first timestamp (float seconds) from rgb.txt of a TUM sequence."""
    rgb_txt = os.path.join(seq_dir, "rgb.txt")
    if not os.path.exists(rgb_txt):
        return None
    with open(rgb_txt) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                return float(line.split()[0])
    return None


def _seq_dir(seq):
    """Return dataset base directory for a sequence name."""
    tum = f"/home/ros/datasets/TUM/{seq}"
    if os.path.isdir(tum):
        return tum
    return None


def _parse_img_ts_ns(fname):
    """Parse image filename stem → nanoseconds.
    TUM:   '1305031452.791720' (decimal seconds) → int(float * 1e9)
    EuRoC: '1403636579763555584' (integer ns)    → int
    """
    stem = os.path.splitext(fname)[0]
    if '.' in stem:
        return int(float(stem) * 1e9)
    return int(stem)


def _find_image(img_dir, ts_ns, tol_ns=50_000_000):
    """Find closest .png in img_dir to ts_ns within tol_ns. Returns path or None."""
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


def _parse_lc_pairs(log_path):
    """Return deduplicated list of (robot_a, kf_a, robot_b, kf_b) inter-robot LCs."""
    pattern = re.compile(
        r"New inter-robot loop closure measurement: \((\d+),(\d+)\) -> \((\d+),(\d+)\)")
    seen, pairs = set(), []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if not m: continue
            ra, ka, rb, kb = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            key = (min((ra, ka), (rb, kb)), max((ra, ka), (rb, kb)))
            if key not in seen:
                seen.add(key)
                pairs.append((ra, ka, rb, kb))
    return pairs


def _load_gt_poses(tum_path):
    """Load groundtruth.tum → (N,4) array [timestamp_sec, x, y, z] sorted by time."""
    data = np.loadtxt(tum_path)
    return data[:, :4]   # timestamp, x, y, z


def _gt_pos_at_ts(gt_poses, ts_sec, tol=0.5):
    """Return xyz of closest GT pose to ts_sec, or None if gap > tol seconds."""
    idx = np.argmin(np.abs(gt_poses[:, 0] - ts_sec))
    if abs(gt_poses[idx, 0] - ts_sec) > tol:
        return None
    return gt_poses[idx, 1:4]


def compute_lc_gt_distances(lc_pairs, ts_map0, ts_map1, gt0, gt1, offset_ns=None):
    """For each accepted LC pair return GT Euclidean distance (m) between matched kf positions.
    offset_ns: {robot_id: ns_to_add} to reverse clock-alignment shift (TUM multi-agent).
    Returns list of floats (pairs with missing timestamps/GT are skipped)."""
    if offset_ns is None:
        offset_ns = {0: 0, 1: 0}
    dists = []
    for ra, ka, rb, kb in lc_pairs:
        if ra == 1: ra, ka, rb, kb = rb, kb, ra, ka   # normalise: robot 0 on left
        ts0 = ts_map0.get(ka)
        ts1 = ts_map1.get(kb)
        if ts0 is None or ts1 is None:
            continue
        pos0 = _gt_pos_at_ts(gt0, (ts0 + offset_ns.get(0, 0)) / 1e9)
        pos1 = _gt_pos_at_ts(gt1, (ts1 + offset_ns.get(1, 0)) / 1e9)
        if pos0 is None or pos1 is None:
            continue
        dists.append(float(np.linalg.norm(pos0 - pos1)))
    return dists


# ── load data ──────────────────────────────────────────────────────────────────

_COLOR_MAP = {
    "baseline":     COLOR_BASELINE,
    "baseline_geo": "#1a9850",   # green — geo-trained CosPlace
    "dinov2":       COLOR_DINOV2,
    "dinov2_real":  "#7b2d8b",   # purple — real-image VLAD
}
methods = {}
for method in METHODS_LIST:
    color = _COLOR_MAP.get(method, COLOR_BASELINE)
    try:
        g2o   = find_latest_g2o(method)
        log   = _latest(method, "ros_launch.log")
        est_r0 = _latest(method, "eval_r0", "estimated.tum")
        gt_r0  = _latest(method, "eval_r0", "groundtruth.tum")
        est_r1 = _latest(method, "eval_r1", "estimated.tum")
        gt_r1  = _latest(method, "eval_r1", "groundtruth.tum")

        ts0, est0, gt0 = sync_and_align(est_r0, gt_r0)
        ts1, est1, gt1 = sync_and_align(est_r1, gt_r1)
        ape0 = compute_ape(est0, gt0)
        ape1 = compute_ape(est1, gt1)
        gt0_full = load_full_gt(SEQ1) if load_full_gt(SEQ1) is not None else gt0
        gt1_full = load_full_gt(SEQ2) if load_full_gt(SEQ2) is not None else gt1

        verts, odom_e, lc_e = parse_g2o(g2o)
        n_intra, n_inter, n_failed = parse_inter_robot_lc(log)
        n_attempted = n_inter + n_failed
        lc_prec = 100 * n_inter / n_attempted if n_attempted > 0 else float("nan")

        lc_pairs = _parse_lc_pairs(log)
        ts_map0  = _load_ts_map(_latest(method), 0)
        ts_map1  = _load_ts_map(_latest(method), 1)
        gt0_path = _latest(method, "eval_r0", "groundtruth.tum")
        gt1_path = _latest(method, "eval_r1", "groundtruth.tum")
        # TUM player shifts robot i timestamps by (t0_seq0 - t0_seq_i); reverse for GT lookup
        _t0_s0 = _first_ts_from_rgb_txt(_seq_dir(SEQ1)) if _seq_dir(SEQ1) else None
        _t0_s1 = _first_ts_from_rgb_txt(_seq_dir(SEQ2)) if _seq_dir(SEQ2) else None
        _gt_offset_ns = {
            0: 0,
            1: int((_t0_s1 - _t0_s0) * 1e9) if (_t0_s0 and _t0_s1) else 0,
        }
        try:
            gt0_poses = _load_gt_poses(gt0_path)
            gt1_poses = _load_gt_poses(gt1_path)
            lc_gt_dists = compute_lc_gt_distances(lc_pairs, ts_map0, ts_map1,
                                                   gt0_poses, gt1_poses, _gt_offset_ns)
        except Exception as _e:
            print(f"  WARNING: GT distance computation failed ({_e})")
            lc_gt_dists = []

        methods[method] = {
            "color": color, "g2o": g2o, "run_dir": _latest(method),
            "ts0": ts0, "est0": est0, "gt0": gt0, "gt0_full": gt0_full, "ape0": ape0,
            "ts1": ts1, "est1": est1, "gt1": gt1, "gt1_full": gt1_full, "ape1": ape1,

            "verts": verts, "odom": odom_e, "lc": lc_e,
            "n_intra": n_intra, "n_inter": n_inter,
            "n_failed": n_failed, "lc_prec": lc_prec,
            "lc_pairs": lc_pairs, "lc_gt_dists": lc_gt_dists,
        }
        tp1  = sum(d <= 1.0 for d in lc_gt_dists)
        tp2  = sum(d <= 2.0 for d in lc_gt_dists)
        tp5  = sum(d <= 5.0 for d in lc_gt_dists)
        n_d  = len(lc_gt_dists)
        print(f"{method}: R0 RMSE={ape0['rmse']:.4f}m  R1 RMSE={ape1['rmse']:.4f}m  "
              f"inter-LC={n_inter}/{n_attempted} ({lc_prec:.1f}% pass)  "
              f"GT precision: @1m={tp1}/{n_d}  @2m={tp2}/{n_d}  @5m={tp5}/{n_d}")
    except FileNotFoundError as e:
        print(f"WARNING: {method} data not found — {e}")


if not methods:
    print("No multi-agent results found. Run run_all.sh first.")
    raise SystemExit(1)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Trajectories for both robots, both methods
# ══════════════════════════════════════════════════════════════════════════════
print("\nfig_multi1_trajectories …")
fig, axes = plt.subplots(1, len(methods), figsize=(7*len(methods), 5))
if len(methods) == 1:
    axes = [axes]

for ax, (method, D) in zip(axes, methods.items()):
    # Use full GT so the reference path is identical across methods
    ax.plot(D["gt0_full"][:,0], D["gt0_full"][:,1], color=COLOR_GT, lw=1.5, label=f"GT {SEQ1}", ls="-")
    ax.plot(D["gt1_full"][:,0], D["gt1_full"][:,1], color=COLOR_GT, lw=1.5, label=f"GT {SEQ2}", ls="--")
    ax.plot(D["est0"][:,0], D["est0"][:,1], color=COLOR_R0, lw=1.3, ls="--",
            label=f"R0 est (RMSE={D['ape0']['rmse']:.3f}m)")
    ax.plot(D["est1"][:,0], D["est1"][:,1], color=COLOR_R1, lw=1.3, ls=":",
            label=f"R1 est (RMSE={D['ape1']['rmse']:.3f}m)")
    ax.scatter(D["gt0_full"][0,0], D["gt0_full"][0,1], c="limegreen", s=80, zorder=5)
    ax.scatter(D["gt1_full"][0,0], D["gt1_full"][0,1], c="limegreen", s=80, zorder=5, marker="^")
    prec_str = f"{D['lc_prec']:.1f}%" if not np.isnan(D['lc_prec']) else "n/a"
    ax.set(xlabel="X (m)", ylabel="Y (m)",
           title=f"{method.title()} — 2-robot XY\n"
                 f"Inter-LC: {D['n_inter']}/{D['n_inter']+D['n_failed']} ({prec_str} pass)  "
                 f"Intra-LC: {D['n_intra']}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=FONT-2)

fig.suptitle(f"Multi-Agent Swarm-SLAM — {SEQ1} + {SEQ2}", fontsize=FONT+2)
fig.tight_layout()
p = os.path.join(OUT_DIR, "fig_multi1_trajectories.png")
fig.savefig(p, bbox_inches="tight", dpi=150); plt.close(fig)
print(f"  → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — APE comparison: single-robot baseline vs multi-agent, per robot
# ══════════════════════════════════════════════════════════════════════════════
print("fig_multi2_ape_comparison …")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
metrics = ["rmse", "mean", "median", "max"]
labels  = ["RMSE", "Mean", "Median", "Max"]
x, w = np.arange(len(metrics)), 0.35

for ax, robot_idx, seq_name in [(axes[0], 0, SEQ1), (axes[1], 1, SEQ2)]:
    for offset, (method, D) in zip([-w/2, w/2], methods.items()):
        ape_key = f"ape{robot_idx}"
        vals = [D[ape_key][m] for m in metrics]
        bars = ax.bar(x + offset, vals, w, color=D["color"], alpha=0.85,
                      label=method.title(), edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=FONT-3,
                    color=D["color"], weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("APE Translation Error (m)")
    ax.set_title(f"Robot {robot_idx} — {seq_name}")
    ax.legend(fontsize=FONT-1)

fig.suptitle(f"Multi-Agent APE Comparison (Sim(3) alignment)\n{SEQ1} + {SEQ2}",
             fontsize=FONT+2)
fig.tight_layout()
p = os.path.join(OUT_DIR, "fig_multi2_ape_comparison.png")
fig.savefig(p, bbox_inches="tight", dpi=150); plt.close(fig)
print(f"  → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Inter-robot vs intra-robot loop closure counts
# ══════════════════════════════════════════════════════════════════════════════
print("fig_multi3_lc_counts …")
fig, ax = plt.subplots(figsize=(7, 4.5))
mnames = list(methods.keys())
x = np.arange(len(mnames))
w = 0.35

intra_vals    = [methods[m]["n_intra"]  for m in mnames]
inter_vals    = [methods[m]["n_inter"]  for m in mnames]
failed_vals   = [methods[m]["n_failed"] for m in mnames]
attempted_vals = [methods[m]["n_inter"] + methods[m]["n_failed"] for m in mnames]

w = 0.25
bars_intra  = ax.bar(x - w, intra_vals, w, label="Intra-robot LC (accepted)",
                     color=[methods[m]["color"] for m in mnames], alpha=0.85, edgecolor="white")
bars_inter  = ax.bar(x,     inter_vals, w, label="Inter-robot LC (accepted)",
                     color=[methods[m]["color"] for m in mnames], alpha=0.55,
                     edgecolor=[methods[m]["color"] for m in mnames], linewidth=1.5)
bars_failed = ax.bar(x + w, failed_vals, w, label="Inter-robot LC (failed geom.)",
                     color="#bbbbbb", alpha=0.7, edgecolor="#999999", linewidth=1.0)

for bar, val in (list(zip(bars_intra, intra_vals)) +
                 list(zip(bars_inter, inter_vals)) +
                 list(zip(bars_failed, failed_vals))):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            str(val), ha="center", va="bottom", fontsize=FONT-1)

# Annotate success % above each inter-robot pair
for i, m in enumerate(mnames):
    prec = methods[m]["lc_prec"]
    label = f"{prec:.1f}%" if not np.isnan(prec) else "n/a"
    mid = (bars_inter[i].get_x() + bars_inter[i].get_width()/2 +
           bars_failed[i].get_x() + bars_failed[i].get_width()/2) / 2
    top = max(inter_vals[i], failed_vals[i]) + 2.5
    ax.text(mid, top, f"pass {label}", ha="center", va="bottom",
            fontsize=FONT-1, color="dimgray", style="italic")

ax.set_xticks(x); ax.set_xticklabels([m.title() for m in mnames])
ax.set_ylabel("Loop Closure Count")
ax.set_title(f"Intra- vs Inter-Robot Loop Closures\n{SEQ1} + {SEQ2}")
ax.legend(fontsize=FONT-1)
fig.tight_layout()
p = os.path.join(OUT_DIR, "fig_multi3_lc_counts.png")
fig.savefig(p, bbox_inches="tight", dpi=150); plt.close(fig)
print(f"  → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Pose graphs for both methods (multi-robot)
# ══════════════════════════════════════════════════════════════════════════════
print("fig_multi4_pose_graph …")
fig, axes = plt.subplots(1, len(methods), figsize=(7*len(methods), 5))
if len(methods) == 1:
    axes = [axes]

for ax, (method, D) in zip(axes, methods.items()):
    verts, odom_e, lc_e = D["verts"], D["odom"], D["lc"]
    ids = sorted(verts)
    xs = [verts[v][0] for v in ids]; ys = [verts[v][1] for v in ids]
    for a, b in odom_e:
        if a in verts and b in verts:
            ax.plot([verts[a][0], verts[b][0]], [verts[a][1], verts[b][1]],
                    color="#aaaaaa", lw=0.5, zorder=1)
    for a, b in lc_e:
        if a in verts and b in verts:
            ax.plot([verts[a][0], verts[b][0]], [verts[a][1], verts[b][1]],
                    color=D["color"], lw=1.5, alpha=0.7, zorder=2)
    ax.scatter(xs, ys, c="#444444", s=5, zorder=3)
    prec_str = f"{D['lc_prec']:.1f}%" if not np.isnan(D['lc_prec']) else "n/a"
    ax.set(xlabel="X (m)", ylabel="Y (m)",
           title=f"{method.title()} — {len(verts)} vertices\n"
                 f"{len(lc_e)} LC edges  "
                 f"({D['n_inter']}/{D['n_inter']+D['n_failed']} inter-robot, {prec_str} pass)")
    ax.set_aspect("equal", adjustable="datalim")

fig.suptitle(f"Multi-Agent Pose Graph — {SEQ1} + {SEQ2}", fontsize=FONT+2)
fig.tight_layout()
p = os.path.join(OUT_DIR, "fig_multi4_pose_graph.png")
fig.savefig(p, bbox_inches="tight", dpi=150); plt.close(fig)
print(f"  → {p}")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Inter-robot LC image pairs (one figure per method)
# ══════════════════════════════════════════════════════════════════════════════
img_dir0, _ = _img_dir(SEQ1)
img_dir1, _ = _img_dir(SEQ2)

if img_dir0 is None or img_dir1 is None:
    print("WARNING: dataset images not found — skipping LC image pair figures")
else:
    # Compute per-robot timestamp offset: tum_player shifts robot i's timestamps
    # by (t0_seq0 - t0_seq_i) so all robots share the same clock base.
    # Reverse it to recover original image timestamps.
    seq_dir0 = _seq_dir(SEQ1)
    seq_dir1 = _seq_dir(SEQ2)
    t0_seq0 = _first_ts_from_rgb_txt(seq_dir0) if seq_dir0 else None
    t0_seq1 = _first_ts_from_rgb_txt(seq_dir1) if seq_dir1 else None
    # offset_ns[robot] = ns to ADD to published ts to get original image ts
    offset_ns = {
        0: 0,
        1: int((t0_seq1 - t0_seq0) * 1e9) if (t0_seq0 and t0_seq1) else 0,
    }

    for method, D in methods.items():
        print(f"fig_multi5_lc_images_{method} …")
        lc_pairs = D["lc_pairs"]
        if not lc_pairs:
            print(f"  No LC pairs for {method} — skipping")
            continue

        ts_map0 = _load_ts_map(D["run_dir"], 0)
        ts_map1 = _load_ts_map(D["run_dir"], 1)

        pairs_found = []
        for ra, ka, rb, kb in lc_pairs:
            if ra == 1:
                ra, ka, rb, kb = rb, kb, ra, ka
            ts0 = ts_map0.get(ka)
            ts1 = ts_map1.get(kb)
            if ts0 is None or ts1 is None:
                continue
            # Reverse player timestamp offset to get original dataset timestamp
            p0 = _find_image(img_dir0, ts0 + offset_ns[0])
            p1 = _find_image(img_dir1, ts1 + offset_ns[1])
            if p0 is None or p1 is None:
                continue
            pairs_found.append((ka, kb, p0, p1))

        if not pairs_found:
            print(f"  WARNING: no images matched for {method} — skipping")
            continue

        MAX_SHOW = min(len(pairs_found), 12)
        N_COLS   = 4   # 2 panels per row × 2 images per panel
        N_ROWS   = int(np.ceil(MAX_SHOW / 2))

        fig = plt.figure(figsize=(16, N_ROWS * 3.5))
        gs  = gridspec.GridSpec(N_ROWS, N_COLS, figure=fig, hspace=0.5, wspace=0.08)

        for i, (ka, kb, p0, p1) in enumerate(pairs_found[:MAX_SHOW]):
            row      = i // 2
            col_base = (i % 2) * 2

            img0 = cv2.imread(p0, cv2.IMREAD_GRAYSCALE)
            img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)

            ax0 = fig.add_subplot(gs[row, col_base])
            ax1 = fig.add_subplot(gs[row, col_base + 1])

            ax0.imshow(img0, cmap="gray", vmin=0, vmax=255)
            ax1.imshow(img1, cmap="gray", vmin=0, vmax=255)

            ax0.set_title(f"R0 kf={ka}\n{SEQ1}", fontsize=8, pad=2)
            ax1.set_title(f"R1 kf={kb}\n{SEQ2}", fontsize=8, pad=2)

            for ax, color in [(ax0, "tomato"), (ax1, "steelblue")]:
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2.5)
                ax.set_xticks([]); ax.set_yticks([])

            fig.text(
                (col_base + 1) / N_COLS,
                1 - (row + 0.58) / N_ROWS,
                f"LC #{i+1}",
                ha="center", va="center", fontsize=7,
                color="gray", style="italic",
                transform=fig.transFigure,
            )

        fig.suptitle(
            f"Inter-Robot LC Image Pairs — {method.title()}\n"
            f"Red border = R0 ({SEQ1})   Blue border = R1 ({SEQ2})\n"
            f"{MAX_SHOW} of {len(lc_pairs)} unique inter-robot LCs shown",
            fontsize=10, y=1.01,
        )

        p = os.path.join(OUT_DIR, f"fig_multi5_lc_images_{method}.png")
        fig.savefig(p, bbox_inches="tight", dpi=130)
        plt.close(fig)
        print(f"  → {p}")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Inter-robot LC GT distance distribution + precision@threshold
# ══════════════════════════════════════════════════════════════════════════════
print("fig_multi6_lc_gt_precision …")
THRESHOLDS  = [1.0, 2.0, 5.0]
THRESH_COLORS = ["#2ca02c", "#ff7f0e", "#d62728"]   # green / orange / red

fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4.5), squeeze=False)
axes = axes[0]

for ax, (method, D) in zip(axes, methods.items()):
    dists = D["lc_gt_dists"]
    n_d = len(dists)

    if n_d == 0:
        ax.text(0.5, 0.5, "No GT distances available",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title(f"{method.title()} — GT LC Precision")
        continue

    # Histogram of raw distances
    max_d = max(dists) * 1.1 if dists else 6.0
    bins = np.linspace(0, max(max_d, 5.5), 20)
    ax.hist(dists, bins=bins, color=D["color"], alpha=0.6, edgecolor="white")

    # Vertical threshold lines + TP annotation
    for thr, col in zip(THRESHOLDS, THRESH_COLORS):
        tp = sum(d <= thr for d in dists)
        pct = 100 * tp / n_d
        ax.axvline(thr, color=col, lw=1.8, ls="--")
        ax.text(thr + 0.05, ax.get_ylim()[1] * 0.95,
                f"@{thr:.0f}m\n{tp}/{n_d}\n({pct:.0f}%)",
                color=col, fontsize=FONT-2, va="top")

    ax.set_xlabel("GT Distance between matched keyframes (m)")
    ax.set_ylabel("Count")
    mean_d = np.mean(dists)
    median_d = np.median(dists)
    ax.set_title(f"{method.title()} — Inter-Robot LC GT Distances\n"
                 f"n={n_d}  mean={mean_d:.2f}m  median={median_d:.2f}m")

fig.suptitle(f"Inter-Robot LC Precision (GT-based) — {SEQ1} + {SEQ2}\n"
             f"Distance = Euclidean between matched keyframe GT positions",
             fontsize=FONT + 1)
fig.tight_layout()
p = os.path.join(OUT_DIR, "fig_multi6_lc_gt_precision.png")
fig.savefig(p, bbox_inches="tight", dpi=150); plt.close(fig)
print(f"  → {p}")

# ── Print precision summary ────────────────────────────────────────────────────
print("\n" + "="*60)
print(f" Inter-Robot LC Precision Summary — {SEQ1} + {SEQ2}")
print("="*60)
print(f"{'Method':<12} {'Accepted':>8} {'Attempted':>10}  {'Pass%':>6}  "
      f"{'TP@1m':>7}  {'TP@2m':>7}  {'TP@5m':>7}  {'Mean dist':>10}")
for method, D in methods.items():
    dists = D["lc_gt_dists"]
    n_d   = len(dists)
    n_acc = D["n_inter"]
    n_att = D["n_inter"] + D["n_failed"]
    prec  = D["lc_prec"]
    tp1   = sum(d <= 1.0 for d in dists)
    tp2   = sum(d <= 2.0 for d in dists)
    tp5   = sum(d <= 5.0 for d in dists)
    mean_d = np.mean(dists) if dists else float("nan")
    def _fmt(tp, n): return f"{tp}/{n}({100*tp/n:.0f}%)" if n else "n/a"
    print(f"{method:<12} {n_acc:>8} {n_att:>10}  {prec:>5.1f}%  "
          f"{_fmt(tp1,n_d):>10}  {_fmt(tp2,n_d):>10}  {_fmt(tp5,n_d):>10}  "
          f"{mean_d:>8.2f}m")
print("="*60)

print(f"\nAll multi-agent figures → {OUT_DIR}/")
