# Swarm-SLAM + AnyLoc: DINOv2-Based Visual Place Recognition for Multi-Robot C-SLAM

This repository is a fork of [MISTLab/Swarm-SLAM](https://github.com/MISTLab/Swarm-SLAM), extended with [AnyLoc](https://anyloc.github.io/) (DINOv2 + VLAD) as a pluggable visual place recognition (VPR) backend. The work was conducted as a course project for CMU 16-833 (Robot Localization and Mapping).

## Overview

[Swarm-SLAM](https://ieeexplore.ieee.org/document/10321649) is a sparse, decentralized Collaborative SLAM (C-SLAM) framework for multi-robot systems. Robots share compact global descriptors to detect inter-robot loop closures and jointly optimize a shared pose graph via decentralized PGO (GTSAM). It supports lidar, stereo, and RGB-D sensing.

This fork introduces AnyLoc's DINOv2 + VLAD descriptors as an alternative VPR backend alongside the default CosPlace (ResNet-18, 64-dim) baseline, and evaluates the accuracy–bandwidth trade-off across descriptor dimensionalities.

## What's New in This Fork

- **AnyLoc/DINOv2 VPR backend** (`src/cslam/cslam/vpr/anyloc.py`) — DINOv2 features aggregated via VLAD with optional PCA compression, served over ZMQ from a standalone descriptor server (`experiments/dinov2_server.py`)
- **EuRoC dataset support** — ASL-format player without rosbag conversion (`src/cslam_experiments/launch/sensors/euroc_asl_player.py`)
- **Evaluation pipeline** — APE trajectory error, loop closure recall, and descriptor bandwidth analysis (`evaluate.py`, `experiments/analyze_lc.py`)
- **Docker environment** — GPU-enabled container with all dependencies pinned for reproducibility

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│  Each Robot                                         │
│  ┌──────────┐   stereo   ┌──────────────────────┐  │
│  │ RTAB-Map │ ─────────► │  VPR Backend         │  │
│  │ Odometry │            │  CosPlace / AnyLoc   │  │
│  └──────────┘            └──────────┬───────────┘  │
│                                     │ descriptor    │
│                          ┌──────────▼───────────┐  │
│                          │  Loop Closure        │  │
│                          │  Detection + MAC     │  │
│                          └──────────┬───────────┘  │
└─────────────────────────────────────┼───────────────┘
                                      │ inter-robot LCs
                          ┌───────────▼──────────┐
                          │  Decentralized PGO   │
                          │  (GTSAM)             │
                          └──────────────────────┘
```

## VPR Backends

| Backend | Descriptor Dim | Notes |
|---------|---------------|-------|
| CosPlace (baseline) | 64 | ResNet-18, default |
| AnyLoc/DINOv2 | 64 / 128 / 256 / 512 | DINOv2 + VLAD + PCA, configurable |

## Repository Structure

```
Swarm-SLAM/
├── src/
│   ├── cslam/               # Core SLAM package (Python + C++)
│   ├── cslam_interfaces/    # Custom ROS 2 message definitions
│   ├── cslam_experiments/   # Launch files and configs
│   └── cslam_visualization/ # Optional RViz2 visualization
├── experiments/             # Standalone scripts (DINOv2 server, analysis)
├── data/                    # Datasets (EuRoC, S3E)
├── results/                 # Experiment outputs
└── docker/                  # Dockerfile and docker-compose
```

## Quick Start

All development and execution runs inside Docker.

```bash
# Build the image (one-time, ~10–15 min)
cd docker && make build

# Start the container
make run

# Inside the container — build the workspace
cd /root/ws
colcon build && source install/setup.bash

# Run a 2-robot EuRoC experiment (CosPlace baseline)
ros2 launch cslam_experiments euroc_stereo.launch.py \
  dataset_path:=/root/datasets/EuRoC \
  sequences:=MH_01_easy,MH_02_easy \
  max_nb_robots:=2

# Evaluate results
python3 /root/ws/src/swarm_slam/evaluate.py
```

For the AnyLoc/DINOv2 backend, start the descriptor server in a separate terminal before launching:

```bash
python3 experiments/dinov2_server.py --port 5555 --device cuda --dim 128
```

Then launch with `config_file:=euroc_stereo_anyloc.yaml`.

## Citation

If you use this work, please cite the original Swarm-SLAM paper:

```bibtex
@ARTICLE{lajoieSwarmSLAM,
  author={Lajoie, Pierre-Yves and Beltrame, Giovanni},
  journal={IEEE Robotics and Automation Letters},
  title={Swarm-SLAM: Sparse Decentralized Collaborative Simultaneous Localization and Mapping Framework for Multi-Robot Systems},
  year={2024},
  volume={9},
  number={1},
  pages={475-482},
  doi={10.1109/LRA.2023.3333742}
}
```