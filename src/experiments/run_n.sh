#!/usr/bin/env bash
# run_n.sh — Run experiments N times for reproducibility averaging.
#
# Usage:
#   bash experiments/run_n.sh <euroc|tum> <seq1> <seq2> <n_times> <gpu_id> [pca_dim]
#
# gpu_id maps to DINOv2 server port: 0->5555, 1->5556, 2->5557, 3->5558
# pca_dim: PCA output dimension for DINOv2 VLAD descriptors (default: 128)
#          Use 64 to match CosPlace baseline dimensionality.
#
# Example:
#   bash experiments/run_n.sh tum rgbd_dataset_freiburg1_desk rgbd_dataset_freiburg1_desk2 3 0
#   bash experiments/run_n.sh tum rgbd_dataset_freiburg1_desk rgbd_dataset_freiburg1_desk2 3 0 64
#   bash experiments/run_n.sh euroc MH_01_easy MH_02_easy 5 1 128

set -eo pipefail

SW=$(cd "$(dirname "$0")/.." && pwd)
EXPERIMENTS="$SW/experiments"

if [ -z "${1}" ] || [ -z "${2}" ] || [ -z "${3}" ] || [ -z "${4}" ] || [ -z "${5}" ]; then
    echo "Usage: bash experiments/run_n.sh <euroc|tum> <seq1> <seq2> <n_times> <gpu_id> [pca_dim]"
    echo "  gpu_id:  0->port 5555, 1->5556, 2->5557, 3->5558"
    echo "  pca_dim: PCA output dimension (default: 128; use 64 to match CosPlace)"
    echo "Example: bash experiments/run_n.sh tum rgbd_dataset_freiburg1_desk rgbd_dataset_freiburg1_desk2 3 0 64"
    exit 1
fi

DATASET=$1
SEQ1=$2
SEQ2=$3
N=$4
GPU_ID=$5
PCA_DIM=${6:-128}

case "$DATASET" in
    euroc) SCRIPT="$EXPERIMENTS/run_all.sh" ;;
    tum)   SCRIPT="$EXPERIMENTS/run_tum_all.sh" ;;
    *)
        echo "ERROR: dataset must be 'euroc' or 'tum', got '$DATASET'"
        exit 1
        ;;
esac

echo "========================================================"
echo " Swarm-SLAM — Repeated Experiments"
echo " Dataset  : $DATASET"
echo " Sequences: $SEQ1 + $SEQ2"
echo " Runs     : $N   GPU: $GPU_ID   PCA dim: $PCA_DIM"
echo "========================================================"

for i in $(seq 1 "$N"); do
    echo ""
    echo "###################################################"
    echo "  RUN $i / $N"
    echo "###################################################"
    bash "$SCRIPT" "$SEQ1" "$SEQ2" "$GPU_ID" "$PCA_DIM"
done

echo ""
echo "========================================================"
echo " All $N runs complete."
echo " Results: $SW/results/$DATASET/multi_agent/${SEQ1}_${SEQ2}/"
echo "========================================================"
