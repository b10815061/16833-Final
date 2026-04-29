#!/usr/bin/env bash
# run_all.sh — Run baseline + DINOv2 experiments for multi-agent EuRoC.
#
# Usage:
#   bash experiments/run_all.sh <seq1> <seq2> <gpu_id> [pca_dim]
#
# gpu_id maps to DINOv2 server port: 0->5555, 1->5556, 2->5557, 3->5558
# pca_dim: PCA output dimension for DINOv2 VLAD descriptors (default: 128)
#
# Example:
#   bash experiments/run_all.sh MH_01_easy MH_02_easy 0
#   bash experiments/run_all.sh MH_01_easy MH_02_easy 0 64

set -eo pipefail

SW=$(cd "$(dirname "$0")/.." && pwd)   # src/swarm_slam/
EXPERIMENTS="$SW/experiments"

if [ -z "${1}" ] || [ -z "${2}" ] || [ -z "${3}" ]; then
    echo "Usage: bash experiments/run_all.sh <seq1> <seq2> <gpu_id> [pca_dim]"
    echo "  gpu_id:  0->port 5555, 1->5556, 2->5557, 3->5558"
    echo "  pca_dim: PCA output dimension (default: 128)"
    echo "Example: bash experiments/run_all.sh MH_01_easy MH_02_easy 0 64"
    exit 1
fi

SEQUENCE=$1
SEQUENCE2=$2
GPU_ID=$3
PCA_DIM=${4:-128}
PORT=$((5555 + GPU_ID))
RATE=0.5
SERVER_LOG="/tmp/dinov2_server_$$.log"
export SWARM_RUNS_FILE="/tmp/swarm_slam_runs_$$.txt"
rm -f "$SWARM_RUNS_FILE"

echo "========================================================"
echo " Swarm-SLAM EuRoC Experiments"
echo " Sequence : $SEQUENCE + $SEQUENCE2   Rate: ${RATE}x   PCA dim: $PCA_DIM"
echo "========================================================"

# ── Experiment 1: Baseline ────────────────────────────────────────────────────
# echo ""
# echo ">>> [1/2] Baseline (CosPlace/ResNet18-ImageNet)"
# bash "$EXPERIMENTS/run_experiment.sh" \
#     "$EXPERIMENTS/euroc_baseline.yaml" baseline "$SEQUENCE" "$RATE"

# # ── Experiment 2: DINOv2 ─────────────────────────────────────────────────────
# echo ""
# echo ">>> [2/2] DINOv2 (dinov2_vitb14 + VLAD-32 + PCA-128)"

# python3 "$EXPERIMENTS/dinov2_server.py" \
#     --port "$PORT" --device "cuda:$GPU_ID" --model dinov2_vitb14 \
#     --layer 9 --clusters 32 --dim 128 \
#     --cache-dir /tmp/dinov2_cache_gpu${GPU_ID} \
#     > "$SERVER_LOG" 2>&1 &
# DINOV2_PID=$!
# echo "DINOv2 server PID=$DINOV2_PID  port=$PORT  gpu=$GPU_ID  log=$SERVER_LOG"

# for i in $(seq 1 150); do
#     grep -q "Listening on" "$SERVER_LOG" 2>/dev/null && { echo "Server ready (${i}s)"; break; }
#     sleep 1
# done
# if ! grep -q "Listening on" "$SERVER_LOG" 2>/dev/null; then
#     echo "ERROR: DINOv2 server failed to start. Check $SERVER_LOG"; kill "$DINOV2_PID" 2>/dev/null; exit 1
# fi
# cp "$SERVER_LOG" "$SW/results/$SEQUENCE/dinov2/dinov2_server.log" 2>/dev/null || true

# bash "$EXPERIMENTS/run_experiment.sh" \
#     "$EXPERIMENTS/euroc_dinov2.yaml" dinov2 "$SEQUENCE" "$RATE"

# kill "$DINOV2_PID" 2>/dev/null || true

# # ── Single-robot figures ──────────────────────────────────────────────────────
# echo ""
# echo ">>> Generating single-robot comparison figures …"
# python3 "$EXPERIMENTS/visualize.py" "$SEQUENCE" \
#     || echo "WARNING: visualize.py failed — figures skipped"

# ── Experiment 3: Multi-agent Baseline ───────────────────────────────────────
echo ""
echo ">>> [3/4] Multi-agent Baseline (CosPlace/ResNet18-ImageNet, 2 robots)"
bash "$EXPERIMENTS/run_experiment.sh" \
    "$EXPERIMENTS/euroc_baseline.yaml" baseline \
    "$SEQUENCE,$SEQUENCE2" "$RATE" 2
BASELINE_RUN=$(grep "^baseline:" "$SWARM_RUNS_FILE" 2>/dev/null | tail -1 | cut -d: -f2- || echo "")

# Wait for ROS2 DDS participants from the previous run to fully deregister.
echo "Waiting 15s for DDS cleanup before next experiment..."
sleep 15

# ── Experiment 4: Multi-agent DINOv2 ─────────────────────────────────────────
echo ""
echo ">>> [4/4] Multi-agent DINOv2 (2 robots)"

DINOV2_PID=$(ps aux | grep "dinov2_server.py.*--port $PORT" | grep -v grep | awk '{print $2}' | head -1 || true)
if [ -n "$DINOV2_PID" ]; then
    echo "DINOv2 server already running on port $PORT (PID=$DINOV2_PID) — skipping start"
    SERVER_OWNED=0
else
    python3 "$EXPERIMENTS/dinov2_server.py" \
        --port "$PORT" --device "cuda:$GPU_ID" --model dinov2_vitb14 \
        --layer 9 --clusters 32 --dim "$PCA_DIM" \
        --cache-dir /tmp/dinov2_cache_gpu${GPU_ID}_dim${PCA_DIM} \
        > "$SERVER_LOG" 2>&1 &
    DINOV2_PID=$!
    SERVER_OWNED=1
    echo "DINOv2 server PID=$DINOV2_PID  port=$PORT  gpu=$GPU_ID  log=$SERVER_LOG"

    for i in $(seq 1 600); do
        grep -q "Listening on" "$SERVER_LOG" 2>/dev/null && { echo "Server ready (${i}s)"; break; }
        [ $((i % 30)) -eq 0 ] && echo "  still waiting for server... (${i}s)"
        sleep 1
    done
    if ! grep -q "Listening on" "$SERVER_LOG" 2>/dev/null; then
        echo "ERROR: DINOv2 server failed to start. Check $SERVER_LOG"; kill "$DINOV2_PID" 2>/dev/null; exit 1
    fi
fi

bash "$EXPERIMENTS/run_experiment.sh" \
    "$EXPERIMENTS/euroc_dinov2.yaml" dinov2 \
    "$SEQUENCE,$SEQUENCE2" "$RATE" 2 "$PORT"
DINOV2_RUN=$(grep "^dinov2:" "$SWARM_RUNS_FILE" 2>/dev/null | tail -1 | cut -d: -f2- || echo "")

[ "${SERVER_OWNED:-0}" -eq 1 ] && kill "$DINOV2_PID" 2>/dev/null || true

# ── Multi-agent figures ───────────────────────────────────────────────────────
echo ""
echo ">>> Generating multi-agent comparison figures ..."
python3 "$EXPERIMENTS/visualize_multi.py" "euroc/$SEQUENCE" "euroc/$SEQUENCE2" \
    --runs "baseline:$BASELINE_RUN" "dinov2:$DINOV2_RUN" \
    || echo "WARNING: visualize_multi.py failed — figures skipped"

echo ">>> Generating inter-robot LC image pair figures ..."
for METHOD in baseline dinov2; do
    MRUN_VAR="${METHOD^^}_RUN"
    python3 "$EXPERIMENTS/plot_inter_lc_images.py" "$METHOD" "$SEQUENCE" "$SEQUENCE2" \
        --run-dir "${!MRUN_VAR}" \
        || echo "WARNING: LC image plot failed for $METHOD"
done

# ── Summary ───────────────────────────────────────────────────────────────────
# echo ""
# echo "========================================================"
# echo " SUMMARY — Single-robot ($SEQUENCE)"
# echo "========================================================"
# for LABEL in baseline dinov2; do
#     EVAL="$SW/results/$SEQUENCE/$LABEL/latest/eval_results.txt"
#     echo ""
#     echo "--- $LABEL ---"
#     [ -f "$EVAL" ] && grep -E "rmse|mean|max" "$EVAL" | head -5 \
#                    || echo "(no eval — check $SW/results/$SEQUENCE/$LABEL/latest/)"
# done

echo ""
echo "========================================================"
echo " SUMMARY — Multi-agent ($SEQUENCE + $SEQUENCE2)"
echo "========================================================"
for LABEL in baseline dinov2; do
    echo ""
    echo "--- $LABEL ---"
    SEQ_KEY="${SEQUENCE}_${SEQUENCE2}"
    EVAL_DIR=$(ls -d "$SW/results/euroc/multi_agent/${SEQ_KEY}/multi-agent_${LABEL}_"* \
               2>/dev/null | sort | tail -1)
    for R in 0 1; do
        EVAL="${EVAL_DIR}/eval_results_r${R}.txt"
        SEQ_NAME="$SEQUENCE"; [ "$R" -eq 1 ] && SEQ_NAME="$SEQUENCE2"
        echo "  Robot $R ($SEQ_NAME):"
        [ -f "$EVAL" ] && grep -E "rmse|mean|max" "$EVAL" | head -3 | sed 's/^/    /' \
                       || echo "    (no eval)"
    done
done
echo ""
echo "Results: $SW/results/"
