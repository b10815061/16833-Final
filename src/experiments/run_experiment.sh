#!/usr/bin/env bash
# run_experiment.sh — Run one Swarm-SLAM EuRoC experiment and evaluate it.
#
# Usage (single robot):
#   bash experiments/run_experiment.sh <config_yaml> <label> [sequence] [rate]
#
# Usage (multi-robot):
#   bash experiments/run_experiment.sh <config_yaml> <label> [seq1,seq2,...] [rate] [nb_robots]
#   e.g.: bash run_experiment.sh euroc_baseline.yaml baseline MH_01_easy,MH_02_easy 3.0 2
#
# Results land in:
#   Single:  results/euroc/<sequence>/<label>_<YYYYMMDD_HHMMSS>/
#   Multi:   results/euroc/multi_agent/multi-agent_<label>_<YYYYMMDD_HHMMSS>/
#     ros/              raw ROS g2o + timestamps (written by CSLAM nodes)
#     eval/   or eval_r0/, eval_r1/, ...
#     ros_launch.log
#     eval_results.txt
#     figures/          generated immediately after evaluation

set -eo pipefail

SW=$(cd "$(dirname "$0")/.." && pwd)   # src/swarm_slam/
WS=$(cd "$SW/../.." && pwd)            # colcon workspace root
DATASET=${EUROC_DATASET_PATH:-/home/ros/datasets/EuRoC}
DATASET_TMP=/tmp/euroc   # extracted writable copy (for sequences not on host mount)

CONFIG_YAML=${1:?"Usage: $0 <config_yaml> <label> [sequences] [rate] [nb_robots] [dinov2_port]"}
LABEL=${2:?"Usage: $0 <config_yaml> <label> [sequences] [rate] [nb_robots] [dinov2_port]"}
SEQUENCES_ARG=${3:-MH_01_easy}   # comma-separated for multi-robot
RATE=${4:-3.0}
NB_ROBOTS=${5:-1}
DINOV2_PORT=${6:-5555}
LAUNCH_DELAY=8

# Derive result directory name and per-robot GT paths
TS=$(date +%Y%m%d_%H%M%S)
if [ "$NB_ROBOTS" -gt 1 ]; then
    SEQ_KEY=$(echo "$SEQUENCES_ARG" | tr ',' '_')
    SEQUENCE_DIR="euroc/multi_agent/$SEQ_KEY"
    RUN_DIR="$SW/results/$SEQUENCE_DIR/multi-agent_${LABEL}_$TS"
else
    SEQUENCE_DIR="euroc/single/$SEQUENCES_ARG"
    RUN_DIR="$SW/results/$SEQUENCE_DIR/${LABEL}_$TS"
fi
mkdir -p "$RUN_DIR/ros"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

# Patch log_folder in config to point at this run's ros/ dir, then install
CONFIG_FILENAME=$(basename "$CONFIG_YAML")
TMP_CONFIG="/tmp/cslam_exp_$$.yaml"
sed "s|log_folder:.*|log_folder: \"$RUN_DIR/ros\"|;
     s|server_address:.*tcp://localhost:[0-9]*|server_address: \"tcp://localhost:$DINOV2_PORT|" \
    "$CONFIG_YAML" > "$TMP_CONFIG"
cp "$TMP_CONFIG" "$WS/install/cslam_experiments/share/cslam_experiments/config/$CONFIG_FILENAME"
rm "$TMP_CONFIG"

echo ""
echo "============================================================"
echo " Experiment  : $LABEL  ($TS)"
echo " Sequences   : $SEQUENCES_ARG"
echo " Robots      : $NB_ROBOTS   Rate: ${RATE}x"
echo " Output      : $RUN_DIR"
echo "============================================================"

# For multi-robot: use a merged symlink directory so both sequences resolve
# under a single dataset_path (host mount is read-only, MH_02_easy lives in /tmp)
if [ "$NB_ROBOTS" -gt 1 ]; then
    LAUNCH_DATASET=/tmp/euroc_merged
    mkdir -p "$LAUNCH_DATASET"
    IFS=',' read -ra _SEQS <<< "$SEQUENCES_ARG"
    for _SEQ in "${_SEQS[@]}"; do
        _SEQ="${_SEQ// /}"
        if [ ! -L "$LAUNCH_DATASET/$_SEQ" ]; then
            if [ -d "$DATASET/$_SEQ/mav0" ]; then
                ln -sfn "$DATASET/$_SEQ" "$LAUNCH_DATASET/$_SEQ"
            else
                ln -sfn "$DATASET_TMP/$_SEQ" "$LAUNCH_DATASET/$_SEQ"
            fi
        fi
    done
else
    LAUNCH_DATASET="$DATASET"
fi

LOG="$RUN_DIR/ros_launch.log"
timeout 600 ros2 launch cslam_experiments euroc_stereo.launch.py \
    max_nb_robots:="$NB_ROBOTS" \
    sequences:="$SEQUENCES_ARG" \
    dataset_path:="$LAUNCH_DATASET" \
    config_file:="$CONFIG_FILENAME" \
    rate:="$RATE" \
    launch_delay_s:="$LAUNCH_DELAY" \
    > "$LOG" 2>&1 || true

echo "ROS launch finished."

# Find the most recent g2o inside this run's ros/ dir
RESULT_DIR=$(find "$RUN_DIR/ros" -name "optimized_global_pose_graph.g2o" \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 \
    | awk '{print $2}' | xargs -r dirname)

if [ -z "$RESULT_DIR" ]; then
    echo "WARNING: No g2o found in $RUN_DIR/ros — evaluation skipped."
    echo "Check $LOG"
    exit 0
fi

echo "Evaluating: $RESULT_DIR"

# For multi-robot: robot i's pose_timestamps{i}.csv lives in experiment_robot_{i}/
# but the g2o is only in experiment_robot_0/. Copy missing timestamp files in.
if [ "$NB_ROBOTS" -gt 1 ]; then
    ROS_DIR="$RUN_DIR/ros"
    for i in $(seq 1 $((NB_ROBOTS - 1))); do
        TS_SRC=$(find "$ROS_DIR" -path "*experiment_robot_${i}*" \
            -name "pose_timestamps${i}.csv" -printf '%T@ %p\n' 2>/dev/null \
            | sort -n | tail -1 | awk '{print $2}')
        if [ -n "$TS_SRC" ] && [ ! -f "$RESULT_DIR/pose_timestamps${i}.csv" ]; then
            cp "$TS_SRC" "$RESULT_DIR/pose_timestamps${i}.csv"
            echo "Copied pose_timestamps${i}.csv from robot ${i} log → $RESULT_DIR"
        fi
    done
fi

# Evaluate each robot against its own sequence ground truth
IFS=',' read -ra SEQ_ARRAY <<< "$SEQUENCES_ARG"
for i in "${!SEQ_ARRAY[@]}"; do
    SEQ="${SEQ_ARRAY[$i]// /}"   # trim whitespace
    # Prefer host-mounted dataset; fall back to writable tmp copy
    if [ -d "$DATASET/$SEQ/mav0" ]; then
        GT_CSV="$DATASET/$SEQ/mav0/state_groundtruth_estimate0/data.csv"
    else
        GT_CSV="$DATASET_TMP/$SEQ/mav0/state_groundtruth_estimate0/data.csv"
    fi

    if [ "$NB_ROBOTS" -gt 1 ]; then
        EVAL_DIR="$RUN_DIR/eval_r${i}"
        RESULTS_TXT="$RUN_DIR/eval_results_r${i}.txt"
        echo ""
        echo "--- Robot $i ($SEQ) ---"
    else
        EVAL_DIR="$RUN_DIR/eval"
        RESULTS_TXT="$RUN_DIR/eval_results.txt"
    fi

    python3 "$SW/evaluate.py" "$RESULT_DIR" "$GT_CSV" \
        --robot-id "$i" \
        --save "$EVAL_DIR" \
        2>&1 | tee "$RESULTS_TXT" || true
done

# echo ""
# python3 "$SW/experiments/visualize.py" "$RUN_DIR" || true

# Write run dir so parent scripts can reference the exact run for figures.
# SWARM_RUNS_FILE is set by the parent (run_all.sh) to a PID-unique path,
# avoiding collisions when multiple experiments run in parallel.
if [ -n "$SWARM_RUNS_FILE" ]; then
    echo "${LABEL}:${RUN_DIR}" >> "$SWARM_RUNS_FILE"
fi

echo ""
echo "Done → $RUN_DIR"
