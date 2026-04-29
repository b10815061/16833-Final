#!/usr/bin/env python3
"""
lc_gt_error.py  —  Per-edge loop closure accuracy vs ground truth.

For every LC edge (i, j) in the optimised g2o:
  1. Find the timestamp of vertex i and j by matching their pose to estimated.tum
  2. Interpolate ground truth at those two timestamps
  3. Compute the TRUE relative pose T_gt_ij = T_gt_i^-1 * T_gt_j
  4. Compare to the MEASURED relative pose in the g2o edge
  5. Error = |t_measured - t_true|  (translation, metres)
            + angle between measured and true rotation (degrees)

Usage:
    python3 experiments/lc_gt_error.py MH_01_easy
"""

import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 11})

from scipy.spatial.transform import Rotation

# ── paths ─────────────────────────────────────────────────────────────────────
SEQUENCE  = sys.argv[1] if len(sys.argv) > 1 else "MH_01_easy"
SW        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR   = os.path.join(SW, "results", SEQUENCE)
OUT_DIR   = os.path.join(RES_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

COLOR_BASELINE = "#E07B54"
COLOR_DINOV2   = "#5B8DB8"

# ── helpers ───────────────────────────────────────────────────────────────────

def _latest_run(method):
    dirs = sorted(glob.glob(os.path.join(RES_DIR, f"{method}_2[0-9]*")))
    return dirs[-1] if dirs else os.path.join(RES_DIR, f"{method}_MISSING")


def find_latest_g2o(method):
    ros_dir = os.path.join(_latest_run(method), "ros")
    best = None
    for root, dirs, files in os.walk(ros_dir):
        for f in files:
            if f == "optimized_global_pose_graph.g2o":
                p = os.path.join(root, f)
                if best is None or os.path.getmtime(p) > os.path.getmtime(best):
                    best = p
    if best is None:
        raise FileNotFoundError(f"No g2o found for {method}")
    return best


def find_estimated_tum(method):
    run_dir = _latest_run(method)
    p = os.path.join(run_dir, "eval", "estimated.tum")
    if not os.path.exists(p):
        for root, dirs, files in os.walk(run_dir):
            if "estimated.tum" in files:
                return os.path.join(root, "estimated.tum")
        raise FileNotFoundError(f"No estimated.tum for {method}")
    return p


def find_groundtruth_tum(method):
    run_dir = _latest_run(method)
    p = os.path.join(run_dir, "eval", "groundtruth.tum")
    if not os.path.exists(p):
        for root, dirs, files in os.walk(run_dir):
            if "groundtruth.tum" in files:
                return os.path.join(root, "groundtruth.tum")
        raise FileNotFoundError(f"No groundtruth.tum for {method}")
    return p


def load_tum(path):
    """Returns (timestamps, poses) where poses[i] = [tx ty tz qx qy qz qw]."""
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1:]


def parse_g2o(path):
    """Returns vertices {id: pose_array} and lc_edges list."""
    vertices = {}
    all_edges = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if parts[0] == "VERTEX_SE3:QUAT":
                vid = int(parts[1])
                pose = np.array([float(x) for x in parts[2:9]])
                vertices[vid] = pose
            elif parts[0] == "EDGE_SE3:QUAT":
                i, j = int(parts[1]), int(parts[2])
                t = np.array([float(x) for x in parts[3:6]])
                q = np.array([float(x) for x in parts[6:10]])  # qx qy qz qw
                all_edges.append({"i": i, "j": j, "t_meas": t, "q_meas": q})
    # LC edges: non-consecutive vertex IDs
    sorted_ids = sorted(vertices.keys())
    id_to_idx  = {vid: idx for idx, vid in enumerate(sorted_ids)}
    lc_edges = []
    for e in all_edges:
        if abs(id_to_idx.get(e["i"], 0) - id_to_idx.get(e["j"], 0)) > 1:
            lc_edges.append(e)
    return vertices, lc_edges


def match_vertex_to_timestamp(vertex_pose, est_ts, est_poses, tol=1e-3):
    """Find the timestamp whose estimated pose best matches the g2o vertex pose."""
    diffs = np.linalg.norm(est_poses[:, :3] - vertex_pose[:3], axis=1)
    idx = np.argmin(diffs)
    if diffs[idx] > tol:
        return None  # no match within tolerance
    return est_ts[idx]


def interpolate_gt(ts, gt_ts, gt_poses):
    """Linear interpolation of GT pose at timestamp ts."""
    if ts <= gt_ts[0]:
        return gt_poses[0]
    if ts >= gt_ts[-1]:
        return gt_poses[-1]
    idx = np.searchsorted(gt_ts, ts)
    t0, t1 = gt_ts[idx-1], gt_ts[idx]
    p0, p1 = gt_poses[idx-1], gt_poses[idx]
    alpha = (ts - t0) / (t1 - t0)
    # linear interp on translation, slerp on rotation
    t_interp = p0[:3] + alpha * (p1[:3] - p0[:3])
    # nlerp on quaternion (good enough for small intervals)
    q_interp = p0[3:] + alpha * (p1[3:] - p0[3:])
    q_interp /= np.linalg.norm(q_interp)
    return np.concatenate([t_interp, q_interp])


def pose_to_matrix(pose):
    """[tx ty tz qx qy qz qw] → 4×4 homogeneous matrix."""
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    return T


def relative_pose_error(t_meas, q_meas, T_gt_ij):
    """
    Compare measured LC relative pose to ground truth relative pose.
    Returns (trans_error_m, rot_error_deg).
    """
    T_meas = np.eye(4)
    T_meas[:3, 3] = t_meas
    T_meas[:3, :3] = Rotation.from_quat(q_meas).as_matrix()

    # error = T_meas^-1 * T_gt
    T_err = np.linalg.inv(T_meas) @ T_gt_ij
    trans_err = np.linalg.norm(T_err[:3, 3])
    rot_err   = Rotation.from_matrix(T_err[:3, :3]).magnitude() * 180 / np.pi
    return trans_err, rot_err


# ── main analysis ─────────────────────────────────────────────────────────────

results = {}

for method, color in [("baseline", COLOR_BASELINE), ("dinov2", COLOR_DINOV2)]:
    g2o_path = find_latest_g2o(method)
    est_path = find_estimated_tum(method)
    gt_path  = find_groundtruth_tum(method)

    vertices, lc_edges = parse_g2o(g2o_path)
    est_ts, est_poses  = load_tum(est_path)
    gt_ts,  gt_poses   = load_tum(gt_path)

    print(f"\n{'='*60}")
    print(f"Method: {method}  |  {len(lc_edges)} LC edges")

    trans_errors, rot_errors = [], []
    skipped = 0

    for e in lc_edges:
        vi = vertices.get(e["i"])
        vj = vertices.get(e["j"])
        if vi is None or vj is None:
            skipped += 1; continue

        ts_i = match_vertex_to_timestamp(vi, est_ts, est_poses)
        ts_j = match_vertex_to_timestamp(vj, est_ts, est_poses)
        if ts_i is None or ts_j is None:
            skipped += 1; continue

        gt_i = interpolate_gt(ts_i, gt_ts, gt_poses)
        gt_j = interpolate_gt(ts_j, gt_ts, gt_poses)

        T_gt_i  = pose_to_matrix(gt_i)
        T_gt_j  = pose_to_matrix(gt_j)
        T_gt_ij = np.linalg.inv(T_gt_i) @ T_gt_j  # true relative pose i→j

        te, re = relative_pose_error(e["t_meas"], e["q_meas"], T_gt_ij)
        trans_errors.append(te)
        rot_errors.append(re)

    trans_errors = np.array(trans_errors)
    rot_errors   = np.array(rot_errors)

    print(f"  Matched: {len(trans_errors)} / {len(lc_edges)}  (skipped {skipped})")
    print(f"\n  Per-edge translation error (measured vs GT):")
    print(f"    Mean:   {trans_errors.mean():.4f} m")
    print(f"    Median: {np.median(trans_errors):.4f} m")
    print(f"    Max:    {trans_errors.max():.4f} m")
    print(f"    <0.1m:  {(trans_errors < 0.1).sum()} ({100*(trans_errors<0.1).mean():.0f}%)")
    print(f"    <0.3m:  {(trans_errors < 0.3).sum()} ({100*(trans_errors<0.3).mean():.0f}%)")
    print(f"    ≥0.5m:  {(trans_errors >= 0.5).sum()} ({100*(trans_errors>=0.5).mean():.0f}%)")
    print(f"\n  Per-edge rotation error:")
    print(f"    Mean:   {rot_errors.mean():.2f}°")
    print(f"    Median: {np.median(rot_errors):.2f}°")
    print(f"    Max:    {rot_errors.max():.2f}°")

    results[method] = {
        "color": color,
        "trans_errors": trans_errors,
        "rot_errors":   rot_errors,
        "n_lc": len(lc_edges),
    }

# ── Fig E: per-edge translation error histogram ───────────────────────────────
print("\nGenerating figE_lc_trans_error.png …")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, (method, R) in zip(axes, results.items()):
    te = R["trans_errors"]
    bins = np.linspace(0, min(te.max() * 1.1, 3.0), 40)
    ax.hist(te, bins=bins, color=R["color"], alpha=0.85, edgecolor="white")
    ax.axvline(np.median(te), color="black", ls="--", lw=1.5,
               label=f"Median: {np.median(te):.3f} m")
    ax.axvline(te.mean(), color="gray", ls=":", lw=1.5,
               label=f"Mean: {te.mean():.3f} m")
    ax.set(xlabel="Translation error vs GT (m)",
           ylabel="Count",
           title=f"{method.title()} — {R['n_lc']} LC edges")
    ax.legend(fontsize=10)

fig.suptitle(f"Per-Edge Loop Closure Translation Error (measured vs ground truth)\nEuRoC {SEQUENCE}",
             fontsize=12)
fig.tight_layout()
pE = os.path.join(OUT_DIR, "figE_lc_trans_error.png")
fig.savefig(pE, bbox_inches="tight", dpi=150); plt.close(fig)
print(f"  → {pE}")

# ── Fig F: per-edge rotation error histogram ──────────────────────────────────
print("Generating figF_lc_rot_error.png …")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, (method, R) in zip(axes, results.items()):
    re = R["rot_errors"]
    bins = np.linspace(0, min(re.max() * 1.1, 30), 40)
    ax.hist(re, bins=bins, color=R["color"], alpha=0.85, edgecolor="white")
    ax.axvline(np.median(re), color="black", ls="--", lw=1.5,
               label=f"Median: {np.median(re):.2f}°")
    ax.axvline(re.mean(), color="gray", ls=":", lw=1.5,
               label=f"Mean: {re.mean():.2f}°")
    ax.set(xlabel="Rotation error vs GT (°)",
           ylabel="Count",
           title=f"{method.title()} — {R['n_lc']} LC edges")
    ax.legend(fontsize=10)

fig.suptitle(f"Per-Edge Loop Closure Rotation Error (measured vs ground truth)\nEuRoC {SEQUENCE}",
             fontsize=12)
fig.tight_layout()
pF = os.path.join(OUT_DIR, "figF_lc_rot_error.png")
fig.savefig(pF, bbox_inches="tight", dpi=150); plt.close(fig)
print(f"  → {pF}")

# ── Fig G: CDF comparison ─────────────────────────────────────────────────────
print("Generating figG_lc_error_cdf.png …")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax_idx, (metric, label, unit) in enumerate([
    ("trans_errors", "Translation error", "m"),
    ("rot_errors",   "Rotation error",    "°"),
]):
    ax = axes[ax_idx]
    for method, R in results.items():
        vals = np.sort(R[metric])
        cdf  = np.arange(1, len(vals)+1) / len(vals)
        ax.plot(vals, cdf, color=R["color"], lw=2, label=method.title())
    ax.set(xlabel=f"{label} ({unit})",
           ylabel="Cumulative fraction",
           title=f"CDF of per-edge {label.lower()}")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

fig.suptitle(f"Loop Closure Constraint Accuracy — Cumulative Distribution\nEuRoC {SEQUENCE}",
             fontsize=12)
fig.tight_layout()
pG = os.path.join(OUT_DIR, "figG_lc_error_cdf.png")
fig.savefig(pG, bbox_inches="tight", dpi=150); plt.close(fig)
print(f"  → {pG}")

print(f"\nAll figures → {OUT_DIR}")
