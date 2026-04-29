#!/usr/bin/env bash
# run_tum_experiment.sh — Run one Swarm-SLAM TUM RGB-D experiment and evaluate.
#
# Usage (single robot):
#   bash experiments/run_tum_experiment.sh <config_yaml> <label> [sequence] [rate]
#
# Usage (multi-robot):
#   bash experiments/run_tum_experiment.sh <config_yaml> <label> [seq1,seq2,...] [rate] [nb_robots]
#
# Results land in:
#   results/tum/<sequence>/<label>/run_<YYYYMMDD_HHMMSS>/

set -eo pipefail

SW=$(cd "$(dirname "$0")/.." && pwd)   # src/swarm_slam/
WS=$(cd "$SW/../.." && pwd)            # colcon workspace root
DATASET=${TUM_DATASET_PATH:-/home/ros/datasets/TUM}

CONFIG_YAML=${1:?"Usage: $0 <config_yaml> <label> [sequences] [rate] [nb_robots] [dinov2_port]"}
LABEL=${2:?"Usage: $0 <config_yaml> <label> [sequences] [rate] [nb_robots] [dinov2_port]"}
SEQUENCES_ARG=${3:-rgbd_dataset_freiburg1_desk}
RATE=${4:-1.0}
NB_ROBOTS=${5:-1}
DINOV2_PORT=${6:-5555}
LAUNCH_DELAY=8

TS=$(date +%Y%m%d_%H%M%S)
if [ "$NB_ROBOTS" -gt 1 ]; then
    SEQ_KEY=$(echo "$SEQUENCES_ARG" | tr ',' '_')
    SEQUENCE_DIR="tum/multi_agent/$SEQ_KEY"
    RUN_DIR="$SW/results/$SEQUENCE_DIR/multi-agent_${LABEL}_$TS"
else
    SEQUENCE_DIR="tum/single/$SEQUENCES_ARG"
    RUN_DIR="$SW/results/$SEQUENCE_DIR/${LABEL}_$TS"
fi
mkdir -p "$RUN_DIR/ros"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

CONFIG_FILENAME=$(basename "$CONFIG_YAML")
TMP_CONFIG="/tmp/cslam_tum_$$.yaml"
sed "s|log_folder:.*|log_folder: \"$RUN_DIR/ros\"|;
     s|server_address:.*tcp://localhost:[0-9]*|server_address: \"tcp://localhost:$DINOV2_PORT|" \
    "$CONFIG_YAML" > "$TMP_CONFIG"
cp "$TMP_CONFIG" "$WS/install/cslam_experiments/share/cslam_experiments/config/$CONFIG_FILENAME"
rm "$TMP_CONFIG"

# Auto-start cosplace_server if the config requests the server-mode technique.
# This offloads ResNet18 inference to a separate process so RTAB-Map odometry
# is not starved of CPU during multi-agent experiments.
COSPLACE_SERVER_PID=""
if grep -q 'global_descriptor_technique.*cosplace_server' "$CONFIG_YAML"; then
    # Kill any stale process already holding port 5556
    fuser -k 5556/tcp 2>/dev/null || true
    sleep 1
    CKPT="$WS/install/cslam/share/cslam/models/resnet18_64_imagenet.pth"
    python3 "$SW/cslam/cslam/cosplace_server.py" --checkpoint "$CKPT" --port 5556 \
        > "$RUN_DIR/cosplace_server.log" 2>&1 &
    COSPLACE_SERVER_PID=$!
    echo "cosplace_server started (PID=$COSPLACE_SERVER_PID), waiting for model load…"
    sleep 5
fi

echo ""
echo "============================================================"
echo " TUM Experiment : $LABEL  ($TS)"
echo " Sequences      : $SEQUENCES_ARG"
echo " Robots         : $NB_ROBOTS   Rate: ${RATE}x"
echo " Output         : $RUN_DIR"
echo "============================================================"

LOG="$RUN_DIR/ros_launch.log"
ROBOT_DELAY_S=0
timeout 600 ros2 launch cslam_experiments tum_rgbd.launch.py \
    max_nb_robots:="$NB_ROBOTS" \
    sequences:="$SEQUENCES_ARG" \
    dataset_path:="$DATASET" \
    config_file:="$CONFIG_FILENAME" \
    rate:="$RATE" \
    launch_delay_s:="$LAUNCH_DELAY" \
    robot_delay_s:="$ROBOT_DELAY_S" \
    > "$LOG" 2>&1 || true

echo "ROS launch finished."

RESULT_DIR=$(find "$RUN_DIR/ros" -name "optimized_global_pose_graph.g2o" \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 \
    | awk '{print $2}' | xargs -r dirname)

if [ -z "$RESULT_DIR" ]; then
    echo "WARNING: No g2o found — evaluation skipped. Check $LOG"
    exit 0
fi

# Copy multi-robot timestamp files
if [ "$NB_ROBOTS" -gt 1 ]; then
    for i in $(seq 1 $((NB_ROBOTS - 1))); do
        TS_SRC=$(find "$RUN_DIR/ros" -path "*experiment_robot_${i}*" \
            -name "pose_timestamps${i}.csv" -printf '%T@ %p\n' 2>/dev/null \
            | sort -n | tail -1 | awk '{print $2}')
        [ -n "$TS_SRC" ] && [ ! -f "$RESULT_DIR/pose_timestamps${i}.csv" ] && \
            cp "$TS_SRC" "$RESULT_DIR/pose_timestamps${i}.csv"
    done
fi

# Evaluate — TUM ground truth is groundtruth.txt (TUM format, not EuRoC CSV)
IFS=',' read -ra SEQ_ARRAY <<< "$SEQUENCES_ARG"

# Compute t0 for the first sequence (the clock reference)
_first_tum_ts() {
    grep -v '^#' "$1/rgb.txt" | awk 'NR==1{print $1; exit}'
}
T0_SEQ0=$(_first_tum_ts "$DATASET/${SEQ_ARRAY[0]// /}")

for i in "${!SEQ_ARRAY[@]}"; do
    SEQ="${SEQ_ARRAY[$i]// /}"
    GT_TXT="$DATASET/$SEQ/groundtruth.txt"

    if [ "$NB_ROBOTS" -gt 1 ]; then
        EVAL_DIR="$RUN_DIR/eval_r${i}"
        RESULTS_TXT="$RUN_DIR/eval_results_r${i}.txt"
        # Robot i's player shifted timestamps by (t0_seq0 - t0_seq_i + robot_delay_s*i);
        # undo that here so pose timestamps match the original TUM ground truth epoch.
        T0_SEQ_I=$(_first_tum_ts "$DATASET/$SEQ")
        TIME_OFFSET=$(python3 -c "print($T0_SEQ_I - $T0_SEQ0 - $ROBOT_DELAY_S * $i)")
    else
        EVAL_DIR="$RUN_DIR/eval"
        RESULTS_TXT="$RUN_DIR/eval_results.txt"
        TIME_OFFSET=0.0
    fi

    python3 "$SW/evaluate.py" "$RESULT_DIR" "$GT_TXT" \
        --robot-id "$i" \
        --gt-format tum \
        --time-offset-s "$TIME_OFFSET" \
        --save "$EVAL_DIR" \
        2>&1 | tee "$RESULTS_TXT" || true
done

[ -n "$COSPLACE_SERVER_PID" ] && kill "$COSPLACE_SERVER_PID" 2>/dev/null || true

# python3 "$SW/experiments/visualize.py" "$RUN_DIR" || true

# Write run dir so parent scripts can reference the exact run for figures.
# SWARM_RUNS_FILE is set by the parent (run_tum_all.sh / run_all.sh) to a
# PID-unique path, avoiding collisions when multiple experiments run in parallel.
if [ -n "$SWARM_RUNS_FILE" ]; then
    echo "${LABEL}:${RUN_DIR}" >> "$SWARM_RUNS_FILE"
fi

echo "Done → $RUN_DIR"
