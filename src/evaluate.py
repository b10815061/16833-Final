#!/usr/bin/env python3
"""Evaluate Swarm-SLAM results against ground truth using evo.

Reads the optimized g2o pose graph and pose timestamps from a CSLAM
experiment, converts to TUM format, aligns with ground truth,
and runs evo_ape for evaluation.

Usage:
  python3 evaluate.py <result_dir> <gt_file> [--gt-format {euroc,tum}] [--plot]

Examples:
  # EuRoC (default):
  python3 evaluate.py results/euroc/MH_01/baseline/latest \
    /home/ros/datasets/EuRoC/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv

  # TUM RGB-D:
  python3 evaluate.py results/tum/rgbd_dataset_freiburg1_desk/baseline/latest \
    /home/ros/datasets/TUM/rgbd_dataset_freiburg1_desk/groundtruth.txt \
    --gt-format tum
"""
import argparse
import csv
import os
import re
import sys
import tempfile
import subprocess
import numpy as np


def decode_g2o_index(vertex_id):
    """Extract keyframe index from GTSAM Symbol (char + index)."""
    return vertex_id & 0xFFFFFFFF


def decode_ts_index(vertex_id):
    """Extract keyframe index from GTSAM LabeledSymbol (label + char + index)."""
    return vertex_id & 0xFFFFFFFF


def robot_char(vertex_id):
    """Extract the robot-identifying character from bits 48-55 of a GTSAM key.

    Swarm-SLAM assigns each robot a character: robot 0 → 'A', robot 1 → 'B', etc.
    Both g2o vertex IDs and pose_timestamps vertex IDs encode this in bits 48-55.
    """
    return (vertex_id >> 48) & 0xFF


def parse_g2o(filepath, robot_id=0):
    """Parse g2o file, return dict of {keyframe_index: (x, y, z, qx, qy, qz, qw)}.

    Only includes vertices belonging to robot_id (identified by bits 48-55 of the
    GTSAM key: robot 0 → 'A'/0x41, robot 1 → 'B'/0x42, etc.).
    """
    target_char = ord('A') + robot_id
    poses = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('VERTEX_SE3:QUAT'):
                parts = line.strip().split()
                vid = int(parts[1])
                if robot_char(vid) != target_char:
                    continue
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                qx, qy, qz, qw = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
                idx = decode_g2o_index(vid)
                poses[idx] = (x, y, z, qx, qy, qz, qw)
    return poses


def parse_timestamps(filepath):
    """Parse pose_timestamps CSV, return dict of {keyframe_index: timestamp_sec}."""
    timestamps = {}
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 3:
                continue
            vid = int(row[0])
            sec = int(row[1])
            nanosec = int(row[2])
            idx = decode_ts_index(vid)
            timestamps[idx] = sec + nanosec * 1e-9
    return timestamps


def parse_euroc_gt(filepath):
    """Parse EuRoC ground truth CSV, return list of (timestamp_sec, x, y, z, qx, qy, qz, qw)."""
    gt = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            ts = int(parts[0]) * 1e-9  # nanoseconds to seconds
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            qw, qx, qy, qz = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            gt.append((ts, x, y, z, qx, qy, qz, qw))
    return gt


def parse_tum_gt(filepath):
    """Parse TUM ground truth file, return list of (timestamp_sec, x, y, z, qx, qy, qz, qw).

    TUM format: timestamp tx ty tz qx qy qz qw (space-separated, float seconds).
    Lines starting with '#' are comments.
    """
    gt = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            ts = float(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            gt.append((ts, x, y, z, qx, qy, qz, qw))
    return gt


def write_tum(filepath, entries):
    """Write TUM format: timestamp x y z qx qy qz qw"""
    with open(filepath, 'w') as f:
        for entry in entries:
            f.write(' '.join(f'{v:.9f}' if i == 0 else f'{v:.6f}' for i, v in enumerate(entry)) + '\n')


def find_closest_gt(gt_entries, target_ts, max_diff=0.05):
    """Find closest ground truth entry by timestamp (within max_diff seconds)."""
    gt_times = np.array([g[0] for g in gt_entries])
    idx = np.searchsorted(gt_times, target_ts)
    best = None
    best_diff = max_diff
    for candidate in [idx - 1, idx]:
        if 0 <= candidate < len(gt_times):
            diff = abs(gt_times[candidate] - target_ts)
            if diff < best_diff:
                best_diff = diff
                best = candidate
    return best


def main():
    parser = argparse.ArgumentParser(description='Evaluate CSLAM results against ground truth')
    parser.add_argument('result_dir', help='Path to result directory (contains g2o + timestamps)')
    parser.add_argument('gt_csv', help='Path to ground truth file (EuRoC CSV or TUM txt)')
    parser.add_argument('--plot', action='store_true', help='Show evo plot')
    parser.add_argument('--save', default=None, help='Save results to this directory')
    parser.add_argument('--robot-id', type=int, default=0, help='Robot ID (default: 0)')
    parser.add_argument('--gt-format', choices=['euroc', 'tum'], default='euroc',
                        help='Ground truth format: euroc (nanosecond CSV) or tum (second-precision txt)')
    parser.add_argument('--time-offset-s', type=float, default=0.0,
                        help='Add this offset (seconds) to pose timestamps before matching GT. '
                             'Use to correct for multi-agent clock alignment (e.g. +73.9 for desk2).')
    args = parser.parse_args()

    # Find files
    g2o_file = os.path.join(args.result_dir, 'optimized_global_pose_graph.g2o')
    ts_file = os.path.join(args.result_dir, f'pose_timestamps{args.robot_id}.csv')

    if not os.path.exists(g2o_file):
        print(f'Error: {g2o_file} not found')
        sys.exit(1)
    if not os.path.exists(ts_file):
        print(f'Error: {ts_file} not found')
        sys.exit(1)

    # Parse
    poses = parse_g2o(g2o_file, robot_id=args.robot_id)
    timestamps = parse_timestamps(ts_file)
    if args.gt_format == 'tum':
        gt = parse_tum_gt(args.gt_csv)
    else:
        gt = parse_euroc_gt(args.gt_csv)

    print(f'Parsed {len(poses)} poses, {len(timestamps)} timestamps, {len(gt)} ground truth entries')

    # Match poses with timestamps by keyframe index
    matched_indices = sorted(set(poses.keys()) & set(timestamps.keys()))
    print(f'Matched {len(matched_indices)} keyframes with both pose and timestamp')

    if len(matched_indices) == 0:
        print('Error: No matching keyframes between g2o and timestamps')
        sys.exit(1)

    # Build TUM entries for estimated trajectory
    est_entries = []
    gt_entries = []
    skipped = 0

    for idx in matched_indices:
        ts = timestamps[idx] + args.time_offset_s
        pose = poses[idx]  # (x, y, z, qx, qy, qz, qw)

        gt_idx = find_closest_gt(gt, ts)
        if gt_idx is None:
            skipped += 1
            continue

        est_entries.append((ts, pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6]))
        gt_entry = gt[gt_idx]
        gt_entries.append((gt_entry[0], gt_entry[1], gt_entry[2], gt_entry[3],
                          gt_entry[4], gt_entry[5], gt_entry[6], gt_entry[7]))

    print(f'Aligned {len(est_entries)} poses ({skipped} skipped, no GT match)')

    if len(est_entries) < 3:
        print('Error: Too few aligned poses for evaluation')
        sys.exit(1)

    # Write TUM files
    save_dir = args.save or tempfile.mkdtemp(prefix='cslam_eval_')
    os.makedirs(save_dir, exist_ok=True)
    est_tum = os.path.join(save_dir, 'estimated.tum')
    gt_tum = os.path.join(save_dir, 'groundtruth.tum')
    write_tum(est_tum, est_entries)
    write_tum(gt_tum, gt_entries)
    print(f'TUM files written to {save_dir}')

    # Run evo_ape
    cmd = [
        'evo_ape', 'tum', gt_tum, est_tum,
        '--align', '--correct_scale',
        '-v',
    ]
    if args.plot:
        cmd.append('--plot')
    if args.save:
        cmd.extend(['--save_results', os.path.join(save_dir, 'results.zip')])

    print(f'\nRunning: {" ".join(cmd)}\n')
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
