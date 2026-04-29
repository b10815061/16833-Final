"""
Swarm-SLAM visualization — generates publication figures for one sequence.

Usage:
    python3 experiments/visualize.py <run_dir>        # per-run figures (called from run_experiment.sh)
    python3 experiments/visualize.py [sequence_path]  # comparison figures across methods

    sequence_path examples: MH_01_easy  euroc/MH_01_easy  tum/rgbd_dataset_freiburg1_desk

Reads from:  results/<sequence>/{baseline,dinov2}_YYYYMMDD_HHMMSS/   (glob, latest of each)
Writes to:   <run_dir>/figures/     (per-run mode)
             results/<sequence>/figures/  (comparison mode)
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import re
import os
import sys
import json
import glob
import subprocess

SW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/swarm_slam/
_arg = sys.argv[1] if len(sys.argv) > 1 else "MH_01_easy"

if os.path.isdir(_arg):
    # Called with a full run_dir: generate figures inside that run
    _run_dir = os.path.realpath(_arg)
    BASE_DIR = os.path.dirname(_run_dir)
    SEQUENCE = os.path.basename(BASE_DIR)
    OUT_DIR  = os.path.join(_run_dir, "figures")
else:
    # Called with a sequence name: comparison figures in sequence-level dir
    # Accepts: "MH_01_easy", "euroc/MH_01_easy", "tum/rgbd_dataset_freiburg1_desk"
    # New layout: results/{dataset}/single/{seq}/
    parts = _arg.split('/')
    if len(parts) == 1:
        # bare sequence name — infer dataset from name
        _dataset = "tum" if _arg.startswith("rgbd") else "euroc"
        _seq = _arg
    else:
        _dataset, _seq = parts[0], parts[-1]
    SEQUENCE = _seq
    BASE_DIR = os.path.join(SW, "results", _dataset, "single", _seq)
    OUT_DIR  = os.path.join(BASE_DIR, "figures")

os.makedirs(OUT_DIR, exist_ok=True)

def _latest(method, *parts):
    dirs = sorted(glob.glob(os.path.join(BASE_DIR, f"*{method}_2[0-9]*")))
    root = dirs[-1] if dirs else os.path.join(BASE_DIR, f"{method}_MISSING")
    return os.path.join(root, *parts)

def _find_latest_g2o(method):
    ros_dir = _latest(method, "ros")
    result = subprocess.run(
        ["find", ros_dir, "-name", "optimized_global_pose_graph.g2o",
         "-printf", "%T@ %p\n"],
        capture_output=True, text=True)
    lines = [l for l in result.stdout.strip().splitlines() if l]
    if not lines:
        raise FileNotFoundError(f"No g2o in {ros_dir}")
    latest = sorted(lines)[-1].split(" ", 1)[1]
    return latest

BASELINE_EST = _latest("baseline", "eval", "estimated.tum")
BASELINE_GT  = _latest("baseline", "eval", "groundtruth.tum")
DINOV2_EST   = _latest("dinov2",   "eval", "estimated.tum")
DINOV2_GT    = _latest("dinov2",   "eval", "groundtruth.tum")
BASELINE_LOG = _latest("baseline", "ros_launch.log")
DINOV2_LOG   = _latest("dinov2",   "ros_launch.log")
BASELINE_G2O = _find_latest_g2o("baseline")
DINOV2_G2O   = _find_latest_g2o("dinov2")


# ── style ─────────────────────────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

PAPER_FONT = 11
plt.rcParams.update({
    "font.size": PAPER_FONT, "axes.titlesize": PAPER_FONT + 1,
    "axes.labelsize": PAPER_FONT, "xtick.labelsize": PAPER_FONT - 1,
    "ytick.labelsize": PAPER_FONT - 1, "legend.fontsize": PAPER_FONT - 1,
    "figure.dpi": 150,
})
COLOR_GT       = "black"
COLOR_BASELINE = "#2166ac"
COLOR_DINOV2   = "#d6604d"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_tum(path):
    ts, xyz = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            ts.append(float(p[0]))
            xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(ts), np.array(xyz)

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

def compute_rpe_via_evo(est_path, gt_path, delta=1):
    """Run evo_rpe and return stats dict {max, mean, median, rmse, std}."""
    try:
        result = subprocess.run(
            ["evo_rpe", "tum", gt_path, est_path,
             "--align", "--correct_scale",
             "--delta", str(delta), "--delta_unit", "f",
             "--save_results", "/tmp/evo_rpe_tmp.zip"],
            capture_output=True, text=True, timeout=60)
        # Parse from stdout
        stats = {}
        for line in result.stdout.splitlines():
            for key in ("max", "mean", "median", "rmse", "std"):
                if line.strip().startswith(key):
                    try:
                        stats[key] = float(line.split()[-1])
                    except ValueError:
                        pass
        return stats if stats else None
    except Exception as e:
        print(f"  WARNING: evo_rpe failed: {e}")
        return None

def parse_loop_closures(log_path):
    pat = re.compile(r"New intra-robot loop closure \((\d+), (\d+)\)")
    idx = []
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                idx.append(max(int(m.group(1)), int(m.group(2))))
    return sorted(idx)

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


# ── load data ─────────────────────────────────────────────────────────────────

print(f"Sequence: {SEQUENCE}  →  {OUT_DIR}")
print("Loading trajectories …")
ts_b, est_b, gt_b = sync_and_align(BASELINE_EST, BASELINE_GT)
ts_d, est_d, gt_d = sync_and_align(DINOV2_EST,   DINOV2_GT)
ape_b = compute_ape(est_b, gt_b)
ape_d = compute_ape(est_d, gt_d)
print(f"  Baseline RMSE: {ape_b['rmse']:.4f} m   DINOv2 RMSE: {ape_d['rmse']:.4f} m")

print("Computing RPE (delta=1 frame) …")
rpe_b = compute_rpe_via_evo(BASELINE_EST, BASELINE_GT, delta=1)
rpe_d = compute_rpe_via_evo(DINOV2_EST,   DINOV2_GT,   delta=1)
if rpe_b:
    print(f"  Baseline RPE  mean={rpe_b.get('mean',float('nan')):.4f}  rmse={rpe_b.get('rmse',float('nan')):.4f}")
if rpe_d:
    print(f"  DINOv2   RPE  mean={rpe_d.get('mean',float('nan')):.4f}  rmse={rpe_d.get('rmse',float('nan')):.4f}")

APE = {"Baseline": ape_b, "DINOv2": ape_d}
t0 = ts_b[0]; time_b = ts_b - t0; time_d = ts_d - t0

print("Loading loop closure logs …")
lc_b = parse_loop_closures(BASELINE_LOG)
lc_d = parse_loop_closures(DINOV2_LOG)
print(f"  Baseline: {len(lc_b)} LCs   DINOv2: {len(lc_d)} LCs")

print("Loading g2o files …")
verts_b, odom_b, lc_edges_b = parse_g2o(BASELINE_G2O)
verts_d, odom_d, lc_edges_d = parse_g2o(DINOV2_G2O)
print(f"  Baseline: {len(verts_b)} verts {len(lc_edges_b)} LC edges")
print(f"  DINOv2:   {len(verts_d)} verts {len(lc_edges_d)} LC edges")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Trajectories
# ══════════════════════════════════════════════════════════════════════════════
print("\nfig1_trajectories …")
fig = plt.figure(figsize=(14, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[2, 2, 1.2], wspace=0.35)
ax_xy, ax_z, ax_leg = [fig.add_subplot(gs[i]) for i in range(3)]
ax_leg.axis("off")

ax_xy.plot(gt_b[:,0],  gt_b[:,1],  color=COLOR_GT,       lw=1.8, label="Ground Truth", zorder=3)
ax_xy.plot(est_b[:,0], est_b[:,1], color=COLOR_BASELINE, lw=1.4, ls="--", label="Baseline (CosPlace)", zorder=4)
ax_xy.plot(est_d[:,0], est_d[:,1], color=COLOR_DINOV2,   lw=1.4, ls=":",  label="DINOv2", zorder=4)
ax_xy.scatter(gt_b[0,0], gt_b[0,1], color="limegreen", s=80, zorder=6, label="Start")
ax_xy.set(xlabel="X (m)", ylabel="Y (m)", title="(A) XY Trajectory (top-down)")
ax_xy.set_aspect("equal", adjustable="datalim")

ax_z.plot(time_b, gt_b[:,2],  color=COLOR_GT,       lw=1.8)
ax_z.plot(time_b, est_b[:,2], color=COLOR_BASELINE, lw=1.4, ls="--")
ax_z.plot(time_d, est_d[:,2], color=COLOR_DINOV2,   lw=1.4, ls=":")
ax_z.set(xlabel="Time (s)", ylabel="Z (m)", title="(B) Altitude vs Time")

handles = [mpatches.Patch(color=c, label=l) for c, l in [
    (COLOR_GT, "Ground Truth"), (COLOR_BASELINE, "Baseline (CosPlace)"),
    (COLOR_DINOV2, "DINOv2"), ("limegreen", "Start marker")]]
ax_leg.legend(handles=handles, loc="upper center", frameon=True)

def _rpe(rpe, key):
    return f"{rpe[key]:.4f}" if rpe and key in rpe else "—"

rows = [["APE RMSE (m)",    f"{ape_b['rmse']:.4f}",   f"{ape_d['rmse']:.4f}"],
        ["APE Mean (m)",    f"{ape_b['mean']:.4f}",   f"{ape_d['mean']:.4f}"],
        ["APE Median (m)",  f"{ape_b['median']:.4f}", f"{ape_d['median']:.4f}"],
        ["APE Max (m)",     f"{ape_b['max']:.4f}",    f"{ape_d['max']:.4f}"],
        ["RPE Mean (m)",    _rpe(rpe_b,"mean"),        _rpe(rpe_d,"mean")],
        ["RPE RMSE (m)",    _rpe(rpe_b,"rmse"),        _rpe(rpe_d,"rmse")],
        ["# KFs",           str(len(verts_b)),         str(len(verts_d))]]
tbl = ax_leg.table(cellText=rows, colLabels=["Metric","Baseline","DINOv2"],
                   loc="center", cellLoc="center", bbox=[0.0,0.08,1.0,0.58])
tbl.auto_set_font_size(False); tbl.set_fontsize(PAPER_FONT - 1)
for (r,c), cell in tbl.get_celld().items():
    if r == 0: cell.set_facecolor("#d0d8e8"); cell.set_text_props(weight="bold")
    elif c == 2: cell.set_facecolor("#fde9e6")
    elif c == 1: cell.set_facecolor("#e6eef8")
ax_leg.set_title("(C) APE Statistics", pad=4)

fig.suptitle(f"EuRoC {SEQUENCE} — Trajectory Comparison (Swarm-SLAM)", fontsize=PAPER_FONT+2, y=1.01)
fig.tight_layout()
p1 = os.path.join(OUT_DIR, "fig1_trajectories.png")
fig.savefig(p1, bbox_inches="tight", dpi=150); plt.close(fig)
print(f"  → {p1}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — APE bar chart
# ══════════════════════════════════════════════════════════════════════════════
print("fig2_ape_comparison …")
metrics = ["max","mean","median","rmse"]
vals_b  = [APE["Baseline"][m] for m in metrics]
vals_d  = [APE["DINOv2"][m]   for m in metrics]
x, w = np.arange(len(metrics)), 0.35

fig, ax = plt.subplots(figsize=(7, 4.5))
bars_b = ax.bar(x-w/2, vals_b, w, color=COLOR_BASELINE, alpha=0.88, label="Baseline (CosPlace)", edgecolor="white")
bars_d = ax.bar(x+w/2, vals_d, w, color=COLOR_DINOV2,   alpha=0.88, label="DINOv2",              edgecolor="white")
for bar, val, col in [(b,v,COLOR_BASELINE) for b,v in zip(bars_b,vals_b)] + \
                     [(b,v,COLOR_DINOV2)   for b,v in zip(bars_d,vals_d)]:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=PAPER_FONT-2, color=col, weight="bold")
for i,(vb,vd) in enumerate(zip(vals_b,vals_d)):
    pct = (vb-vd)/vb*100
    ax.text(x[i], max(vb,vd)+0.035, f"{'+' if pct>0 else ''}{pct:.1f}%",
            ha="center", fontsize=PAPER_FONT-2,
            color="#2e7d32" if pct>0 else "#c62828", style="italic")
ax.set_xticks(x); ax.set_xticklabels(["Max","Mean","Median","RMSE"])
ax.set_ylabel("APE Translation Error (m)")
ax.set_title(f"APE Metric Comparison — Baseline vs DINOv2\n(EuRoC {SEQUENCE}, Sim(3) Umeyama alignment)")
ax.legend(); ax.set_ylim(0, max(vals_b+vals_d)*1.25)
fig.tight_layout()
p2 = os.path.join(OUT_DIR, "fig2_ape_comparison.png")
fig.savefig(p2, bbox_inches="tight", dpi=150); plt.close(fig)
print(f"  → {p2}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Cumulative loop closures
# ══════════════════════════════════════════════════════════════════════════════
print("fig3_loop_closures …")
fig, ax = plt.subplots(figsize=(7, 4))
if lc_b: ax.step(sorted(lc_b), np.arange(1,len(lc_b)+1), where="post",
                 color=COLOR_BASELINE, lw=2, label=f"Baseline (CosPlace) — {len(lc_b)} total")
if lc_d: ax.step(sorted(lc_d), np.arange(1,len(lc_d)+1), where="post",
                 color=COLOR_DINOV2,   lw=2, label=f"DINOv2 — {len(lc_d)} total")
ax.set(xlabel="Keyframe Index", ylabel="Cumulative Loop Closures",
       title=f"Intra-Robot Loop Closures vs Keyframe Index\n(EuRoC {SEQUENCE})")
ax.legend(); fig.tight_layout()
p3 = os.path.join(OUT_DIR, "fig3_loop_closures.png")
fig.savefig(p3, bbox_inches="tight", dpi=150); plt.close(fig)
print(f"  → {p3}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Pose graph
# ══════════════════════════════════════════════════════════════════════════════
print("fig4_pose_graph …")

def draw_pose_graph(ax, vertices, odom_edges, lc_edges, title, color_lc):
    ids = sorted(vertices)
    xs = [vertices[v][0] for v in ids]; ys = [vertices[v][1] for v in ids]
    for a,b in odom_edges:
        if a in vertices and b in vertices:
            ax.plot([vertices[a][0],vertices[b][0]],[vertices[a][1],vertices[b][1]],
                    color="#aaaaaa", lw=0.6, zorder=1)
    for a,b in lc_edges:
        if a in vertices and b in vertices:
            ax.plot([vertices[a][0],vertices[b][0]],[vertices[a][1],vertices[b][1]],
                    color=color_lc, lw=1.5, alpha=0.7, zorder=2)
    ax.scatter(xs, ys, c="#444444", s=6, zorder=3)
    ax.scatter(xs[0],  ys[0],  c="limegreen", s=60, zorder=5, marker="o")
    ax.scatter(xs[-1], ys[-1], c="tomato",     s=60, zorder=5, marker="s")
    ax.legend(handles=[
        plt.Line2D([0],[0], color="#aaaaaa", lw=1.2, label=f"Odom edges ({len(odom_edges)})"),
        plt.Line2D([0],[0], color=color_lc,  lw=2.0, label=f"Loop closures ({len(lc_edges)})"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="limegreen", markersize=7, label="Start"),
        plt.Line2D([0],[0], marker="s", color="w", markerfacecolor="tomato",    markersize=7, label="End"),
    ], fontsize=PAPER_FONT-2)
    ax.set(xlabel="X (m)", ylabel="Y (m)", title=title)
    ax.set_aspect("equal", adjustable="datalim")

fig, (ax_b, ax_d) = plt.subplots(1, 2, figsize=(13, 5))
draw_pose_graph(ax_b, verts_b, odom_b, lc_edges_b,
                f"Baseline (CosPlace)\n{len(verts_b)} vertices, {len(lc_edges_b)} LC edges", COLOR_BASELINE)
draw_pose_graph(ax_d, verts_d, odom_d, lc_edges_d,
                f"DINOv2\n{len(verts_d)} vertices, {len(lc_edges_d)} LC edges", COLOR_DINOV2)
fig.suptitle(f"Optimised Pose Graph — EuRoC {SEQUENCE} (Swarm-SLAM)", fontsize=PAPER_FONT+2)
fig.tight_layout()
p4 = os.path.join(OUT_DIR, "fig4_pose_graph.png")
fig.savefig(p4, bbox_inches="tight", dpi=150); plt.close(fig)
print(f"  → {p4}")

print(f"\nAll figures → {OUT_DIR}/")

# ── Save JSON summary ─────────────────────────────────────────────────────────
summary = {
    "sequence": SEQUENCE,
    "baseline": {
        "ape": ape_b,
        "rpe": rpe_b,
        "loop_closures": len(lc_b),
        "keyframes": len(verts_b),
        "lc_edges": len(lc_edges_b),
    },
    "dinov2": {
        "ape": ape_d,
        "rpe": rpe_d,
        "loop_closures": len(lc_d),
        "keyframes": len(verts_d),
        "lc_edges": len(lc_edges_d),
    },
}
summary_path = os.path.join(OUT_DIR, "summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary JSON → {summary_path}")
