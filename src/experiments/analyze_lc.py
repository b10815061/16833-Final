"""
analyze_lc.py — False-positive loop closure analysis for Swarm-SLAM experiments.

For each method, examines every loop closure edge in the FINAL optimised g2o and
asks: "After the optimizer has done its best, how far apart are these two poses
in 3D space?"

A TRUE loop closure connects two poses at the SAME physical location → distance ≈ 0.
A FALSE POSITIVE connects two different places → the optimizer can't satisfy the
constraint, so one of two things happens:
  (a) The two poses stay far apart — the edge is an obvious inconsistency.
  (b) The optimizer compromises and pulls nearby poses toward each other — this
      *distorts* the trajectory and shows up as higher ATE.

We quantify this by comparing the LC edge's measured relative pose (from the g2o
EDGE line) against the optimised relative pose (difference of vertex positions).
A large residual between these two = the optimizer couldn't satisfy the constraint
= strong evidence of a false positive.

Usage:
    python3 experiments/analyze_lc.py [sequence]

Writes figures to results/<sequence>/figures/
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys, subprocess, glob
from scipy.spatial.transform import Rotation

SW       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQUENCE = sys.argv[1] if len(sys.argv) > 1 else "MH_01_easy"
BASE_DIR = os.path.join(SW, "results", SEQUENCE)
OUT_DIR  = os.path.join(BASE_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

def _latest_run(method):
    dirs = sorted(glob.glob(os.path.join(BASE_DIR, f"{method}_2[0-9]*")))
    return dirs[-1] if dirs else os.path.join(BASE_DIR, f"{method}_MISSING")

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

COLOR_BASELINE = "#2166ac"
COLOR_DINOV2   = "#d6604d"
FONT = 11
plt.rcParams.update({"font.size": FONT, "axes.titlesize": FONT+1,
                     "axes.labelsize": FONT, "figure.dpi": 150})


# ── g2o parsing ───────────────────────────────────────────────────────────────

def find_latest_g2o(method):
    ros_dir = os.path.join(_latest_run(method), "ros")
    r = subprocess.run(["find", ros_dir, "-name", "optimized_global_pose_graph.g2o",
                        "-printf", "%T@ %p\n"], capture_output=True, text=True)
    lines = [l for l in r.stdout.strip().splitlines() if l]
    if not lines:
        raise FileNotFoundError(f"No g2o in {ros_dir}")
    return sorted(lines)[-1].split(" ", 1)[1]


def parse_g2o_full(path):
    """
    Returns:
        vertices : {id: np.array([x,y,z,qx,qy,qz,qw])}
        edges    : list of {i, j, t_meas (xyz), q_meas (xyzw), is_lc}
    """
    vertices = {}
    raw_edges = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p: continue
            if p[0] == "VERTEX_SE3:QUAT":
                vid = int(p[1])
                vals = [float(x) for x in p[2:9]]
                vertices[vid] = np.array(vals)
            elif p[0] == "EDGE_SE3:QUAT":
                i, j = int(p[1]), int(p[2])
                t = np.array([float(p[3]), float(p[4]), float(p[5])])
                q = np.array([float(p[6]), float(p[7]), float(p[8]), float(p[9])])  # xyzw
                raw_edges.append({"i": i, "j": j, "t_meas": t, "q_meas": q})

    id_rank = {v: k for k, v in enumerate(sorted(vertices))}
    for e in raw_edges:
        e["is_lc"] = abs(id_rank.get(e["i"], -999) - id_rank.get(e["j"], -999)) > 1

    return vertices, raw_edges


def pose_distance(v1, v2):
    """Euclidean distance between two vertices (translation only)."""
    return np.linalg.norm(v1[:3] - v2[:3])


def edge_residual(vertices, edge):
    """
    Residual between the measured relative pose (from the g2o EDGE) and
    the optimised relative pose (difference between final vertex positions).

    Returns (trans_residual_m, rot_residual_deg).
    """
    vi = vertices.get(edge["i"])
    vj = vertices.get(edge["j"])
    if vi is None or vj is None:
        return None, None

    # Optimised relative translation: T_i^{-1} * T_j
    t_i, q_i = vi[:3], vi[3:]   # qxyz w
    t_j, q_j = vj[:3], vj[3:]

    R_i = Rotation.from_quat(q_i).as_matrix()
    t_opt = R_i.T @ (t_j - t_i)

    # Measured relative translation from edge
    t_meas = edge["t_meas"]

    trans_res = np.linalg.norm(t_opt - t_meas)

    # Rotation residual
    R_j = Rotation.from_quat(q_j).as_matrix()
    R_opt  = R_i.T @ R_j
    R_meas = Rotation.from_quat(edge["q_meas"]).as_matrix()
    R_diff = R_opt.T @ R_meas
    angle  = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))

    return trans_res, angle


# ── run analysis ──────────────────────────────────────────────────────────────

results = {}
for method, color in [("baseline", COLOR_BASELINE), ("dinov2", COLOR_DINOV2)]:
    g2o = find_latest_g2o(method)
    print(f"\n{'='*60}")
    print(f"Method: {method}   g2o: {g2o}")

    vertices, edges = parse_g2o_full(g2o)
    lc_edges = [e for e in edges if e["is_lc"]]

    # ── 1. Post-optimisation distance between LC endpoints ────────────────
    opt_distances = []
    for e in lc_edges:
        vi = vertices.get(e["i"]); vj = vertices.get(e["j"])
        if vi is not None and vj is not None:
            opt_distances.append(pose_distance(vi, vj))
    opt_distances = np.array(opt_distances)

    # ── 2. Measured translation magnitude per LC edge ─────────────────────
    t_meas_norms = np.array([np.linalg.norm(e["t_meas"]) for e in lc_edges])

    # ── 3. Edge residuals (measured vs optimised relative pose) ───────────
    trans_res, rot_res = [], []
    for e in lc_edges:
        tr, rr = edge_residual(vertices, e)
        if tr is not None:
            trans_res.append(tr); rot_res.append(rr)
    trans_res = np.array(trans_res); rot_res = np.array(rot_res)

    # ── 4. Classify: likely FP if measured translation > 1.5m ────────────
    #   (A true LC = camera revisits the same physical location, so the
    #    PnP-estimated relative translation should be small.  An edge with
    #    |t_meas| >> 1 m was probably matched from a totally different place.)
    T_MEAS_THRESH = 1.5  # m
    fp_mask = t_meas_norms > T_MEAS_THRESH
    n_fp    = fp_mask.sum()
    n_tp    = (~fp_mask).sum()

    print(f"  Total LC edges: {len(lc_edges)}")
    print(f"\n  Post-optimisation distance between LC endpoints:")
    print(f"    mean:   {opt_distances.mean():.3f} m")
    print(f"    median: {np.median(opt_distances):.3f} m")
    print(f"    max:    {opt_distances.max():.3f} m")
    print(f"    <0.5m (same-place revisit):       {(opt_distances < 0.5).sum()} "
          f"({100*(opt_distances<0.5).mean():.0f}%)")
    print(f"    ≥0.5m (different viewpoint / FP): {(opt_distances >= 0.5).sum()} "
          f"({100*(opt_distances>=0.5).mean():.0f}%)")

    print(f"\n  Measured LC relative translation |t_meas|:")
    print(f"    mean:   {t_meas_norms.mean():.3f} m")
    print(f"    median: {np.median(t_meas_norms):.3f} m")
    print(f"    max:    {t_meas_norms.max():.3f} m")
    print(f"    <0.5m:  {(t_meas_norms<0.5).sum()} ({100*(t_meas_norms<0.5).mean():.0f}%)")
    print(f"    0.5–1.5m: {((t_meas_norms>=0.5)&(t_meas_norms<1.5)).sum()} "
          f"({100*((t_meas_norms>=0.5)&(t_meas_norms<1.5)).mean():.0f}%)")
    print(f"    ≥1.5m (large offset — suspect FP): {(t_meas_norms>=1.5).sum()} "
          f"({100*(t_meas_norms>=1.5).mean():.0f}%)")

    print(f"\n  Edge residuals (how well optimizer satisfied each LC):")
    print(f"    Translation residual mean:   {trans_res.mean():.3f} m")
    print(f"    Translation residual median: {np.median(trans_res):.3f} m")
    print(f"    Translation residual max:    {trans_res.max():.3f} m")
    print(f"    Rotation residual mean:      {rot_res.mean():.2f}°")
    print(f"    (all residuals small → GTSAM satisfied every constraint)")

    print(f"\n  Classification by |t_meas| > {T_MEAS_THRESH} m:")
    print(f"    Likely FP: {n_fp} / {len(lc_edges)}  ({100*n_fp/max(len(lc_edges),1):.0f}%)")
    print(f"    Likely TP: {n_tp} / {len(lc_edges)}  ({100*n_tp/max(len(lc_edges),1):.0f}%)")

    results[method] = {
        "color": color, "lc_edges": lc_edges,
        "opt_distances": opt_distances,
        "t_meas_norms": t_meas_norms,
        "trans_res": trans_res, "rot_res": rot_res,
        "fp_mask": fp_mask, "n_fp": n_fp, "n_tp": n_tp,
    }


# ── figures ───────────────────────────────────────────────────────────────────

# ── Fig A: histogram of post-optimisation distances ──────────────────────────
print("\nGenerating figA_lc_distances.png …")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

for ax, (method, R) in zip(axes, results.items()):
    d = R["opt_distances"]
    bins = np.linspace(0, min(d.max()*1.1, 5), 40)
    ax.hist(d, bins=bins, color=R["color"], alpha=0.8, edgecolor="white")
    ax.axvline(0.5, color="red", ls="--", lw=1.5, label="FP threshold (0.5 m)")
    # annotate counts
    n_below = (d < 0.5).sum()
    n_above = (d >= 0.5).sum()
    ax.text(0.02, 0.96, f"≤ 0.5 m (likely TP): {n_below}\n> 0.5 m (likely FP): {n_above}",
            transform=ax.transAxes, va="top", fontsize=FONT-1,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set(xlabel="Distance between LC endpoints after optimisation (m)",
           ylabel="Count",
           title=f"{method.title()} — {len(d)} loop closures")
    ax.legend(fontsize=FONT-1)

fig.suptitle(f"Post-optimisation Distance Between Loop-Closed Poses\nEuRoC {SEQUENCE}",
             fontsize=FONT+2)
fig.tight_layout()
pA = os.path.join(OUT_DIR, "figA_lc_distances.png")
fig.savefig(pA, bbox_inches="tight", dpi=150); plt.close(fig)
print(f"  → {pA}")


# ── Fig B: measured LC translation magnitude histogram ───────────────────────
print("Generating figB_tmeas_distribution.png …")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, (method, R) in zip(axes, results.items()):
    tm = R["t_meas_norms"]
    bins = np.linspace(0, min(tm.max()*1.1, 6), 40)
    ax.hist(tm, bins=bins, color=R["color"], alpha=0.8, edgecolor="white")
    ax.axvline(1.5, color="red", ls="--", lw=1.5, label="FP threshold (1.5 m)")
    n_fp, n_tp = R["n_fp"], R["n_tp"]
    ax.text(0.98, 0.96,
            f"≥1.5m (suspect FP): {n_fp} ({100*n_fp/max(len(tm),1):.0f}%)\n"
            f"<1.5m (likely TP):  {n_tp} ({100*n_tp/max(len(tm),1):.0f}%)",
            transform=ax.transAxes, va="top", ha="right", fontsize=FONT-1,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set(xlabel="Measured relative translation |t_meas| (m)",
           ylabel="Count",
           title=f"{method.title()} — LC measured offset distribution")
    ax.legend(fontsize=FONT-1)

fig.suptitle(f"Loop Closure Measured Relative Translation Magnitude\nEuRoC {SEQUENCE}",
             fontsize=FONT+2)
fig.tight_layout()
pB = os.path.join(OUT_DIR, "figB_tmeas_distribution.png")
fig.savefig(pB, bbox_inches="tight", dpi=150); plt.close(fig)
print(f"  → {pB}")


# ── Fig C: scatter — distance vs residual, colour by TP/FP ──────────────────
print("Generating figC_fp_scatter.png …")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (method, R) in zip(axes, results.items()):
    d  = R["opt_distances"]
    tm = R["t_meas_norms"]
    fp = R["fp_mask"]
    ax.scatter(tm[~fp], d[~fp], c="steelblue", s=30, alpha=0.7,
               label=f"Likely TP ({(~fp).sum()})", zorder=3)
    ax.scatter(tm[fp],  d[fp],  c="tomato",    s=40, alpha=0.9, marker="x",
               label=f"Likely FP ({fp.sum()})", zorder=4)
    ax.axvline(1.5, color="red",  ls="--", lw=1.2, label="FP threshold (1.5 m)")
    lim = max(tm.max(), d.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="gray", ls=":", lw=1, label="y=x")
    ax.set(xlabel="Measured LC relative translation |t_meas| (m)",
           ylabel="Post-optimisation distance between LC endpoints (m)",
           title=f"{method.title()} — measured vs optimised LC offset")
    ax.legend(fontsize=FONT-1)

fig.suptitle(f"Loop Closure: Measured vs Optimised Relative Translation\nEuRoC {SEQUENCE}",
             fontsize=FONT+2)
fig.tight_layout()
pC = os.path.join(OUT_DIR, "figC_fp_scatter.png")
fig.savefig(pC, bbox_inches="tight", dpi=150); plt.close(fig)
print(f"  → {pC}")


# ── Fig D: summary bar — TP vs FP count ──────────────────────────────────────
print("Generating figD_tp_fp_summary.png …")
fig, ax = plt.subplots(figsize=(6, 4))

methods = list(results.keys())
n_tp = [results[m]["n_tp"] for m in methods]
n_fp = [results[m]["n_fp"] for m in methods]
x = np.arange(len(methods))
w = 0.35

b1 = ax.bar(x - w/2, n_tp, w, label="Likely TP", color="steelblue", alpha=0.85)
b2 = ax.bar(x + w/2, n_fp, w, label="Likely FP", color="tomato",    alpha=0.85)

for bar, val in list(zip(b1, n_tp)) + list(zip(b2, n_fp)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(val), ha="center", va="bottom", fontsize=FONT-1, weight="bold")

ax.set_xticks(x)
ax.set_xticklabels([m.title() for m in methods])
ax.set_ylabel("Loop Closure Count")
ax.set_title(f"True vs False Positive Loop Closures\n(threshold: trans residual > 0.5 m)\nEuRoC {SEQUENCE}")
ax.legend()
fig.tight_layout()
pD = os.path.join(OUT_DIR, "figD_tp_fp_summary.png")
fig.savefig(pD, bbox_inches="tight", dpi=150); plt.close(fig)
print(f"  → {pD}")

print(f"\nAll analysis figures → {OUT_DIR}/")
