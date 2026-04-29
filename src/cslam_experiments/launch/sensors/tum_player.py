#!/usr/bin/env python3
"""Plays a TUM RGB-D dataset as ROS 2 topics.

Reads rgb.txt and depth.txt, publishes with proper timestamps and camera_info.
Supports freiburg1 calibration (hardcoded); extend as needed for fr2/fr3.

Usage as a ROS 2 node — parameters:
  dataset_path: path to sequence folder (containing rgb/, depth/, rgb.txt, depth.txt, groundtruth.txt)
  namespace:    robot namespace (e.g., /r0)
  rate:         playback speed multiplier (default 1.0)
  publish_clock: whether to publish /clock (default true)
"""
import os
import time
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from builtin_interfaces.msg import Time as TimeMsg
from sensor_msgs.msg import Image, CameraInfo
from rosgraph_msgs.msg import Clock
from cv_bridge import CvBridge


# ── TUM freiburg1 calibration ─────────────────────────────────────────────────
# From https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
FR1_K = np.array([[517.3, 0.0,   318.6],
                  [0.0,   516.5, 255.3],
                  [0.0,   0.0,   1.0  ]], dtype=np.float64)
FR1_D = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633], dtype=np.float64)
IMG_W, IMG_H = 640, 480
DEPTH_SCALE  = 5000.0  # TUM: depth (uint16) / 5000 = metres


def _ts_to_msg(ts_float):
    msg = TimeMsg()
    msg.sec     = int(ts_float)
    msg.nanosec = int((ts_float - int(ts_float)) * 1e9)
    return msg


def _read_assoc(path):
    """Return list of (timestamp_float, filename_str)."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            entries.append((float(parts[0]), parts[1]))
    return entries


def _associate(rgb_entries, depth_entries, max_dt=0.02):
    """Match RGB and depth by nearest timestamp."""
    pairs = []
    dts = np.array([d[0] for d in depth_entries])
    for ts_rgb, f_rgb in rgb_entries:
        idx = int(np.argmin(np.abs(dts - ts_rgb)))
        if abs(dts[idx] - ts_rgb) <= max_dt:
            pairs.append((ts_rgb, f_rgb, depth_entries[idx][1]))
    return pairs


class TUMPlayer(Node):
    def __init__(self):
        super().__init__('tum_player')
        self.declare_parameter('dataset_path', '')
        self.declare_parameter('namespace', '/r0')
        self.declare_parameter('rate', 1.0)
        self.declare_parameter('publish_clock', True)
        self.declare_parameter('ts_offset_s', 0.0)
        self.declare_parameter('start_skip_s', 0.0)

        self.dataset      = self.get_parameter('dataset_path').value
        self.ns           = self.get_parameter('namespace').value
        self.rate         = self.get_parameter('rate').value
        self.pub_clk      = self.get_parameter('publish_clock').value
        self.ts_offset    = self.get_parameter('ts_offset_s').value
        self.start_skip_s = self.get_parameter('start_skip_s').value

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=10)

        self.pub_rgb   = self.create_publisher(Image,      f'{self.ns}/color/image_raw',                    qos)
        self.pub_depth = self.create_publisher(Image,      f'{self.ns}/aligned_depth_to_color/image_raw',   qos)
        self.pub_info  = self.create_publisher(CameraInfo, f'{self.ns}/color/camera_info',                  qos)
        if self.pub_clk:
            self.pub_clock = self.create_publisher(Clock, '/clock', QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST, depth=1))

        self.bridge = CvBridge()
        self._make_camera_info()

        rgb_entries   = _read_assoc(os.path.join(self.dataset, 'rgb.txt'))
        depth_entries = _read_assoc(os.path.join(self.dataset, 'depth.txt'))
        self.pairs    = _associate(rgb_entries, depth_entries)

        self.get_logger().info(
            f'TUM player: {len(self.pairs)} associated pairs from {self.dataset}')

        self._thread = threading.Thread(target=self._play, daemon=True)
        self._thread.start()

    def _make_camera_info(self):
        msg = CameraInfo()
        msg.width   = IMG_W
        msg.height  = IMG_H
        msg.k       = FR1_K.flatten().tolist()
        msg.d       = FR1_D.tolist()
        msg.r       = np.eye(3).flatten().tolist()
        msg.p       = np.hstack([FR1_K, np.zeros((3,1))]).flatten().tolist()
        msg.distortion_model = 'plumb_bob'
        self._cam_info = msg

    def _publish_frame(self, ts, rgb_path, depth_path):
        ts_msg = _ts_to_msg(ts + self.ts_offset)
        stamp_str = str(ts)

        # RGB
        rgb_img = cv2.imread(os.path.join(self.dataset, rgb_path))
        if rgb_img is None:
            return
        rgb_msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB), 'rgb8')
        rgb_msg.header.stamp = ts_msg
        rgb_msg.header.frame_id = f'{self.ns[1:]}/camera_link'

        # Depth
        depth_img = cv2.imread(os.path.join(self.dataset, depth_path),
                               cv2.IMREAD_ANYDEPTH)
        depth_msg = self.bridge.cv2_to_imgmsg(depth_img, '16UC1')
        depth_msg.header.stamp = ts_msg
        depth_msg.header.frame_id = f'{self.ns[1:]}/camera_link'

        # CameraInfo — ROS 2 messages use __slots__, update header in-place
        self._cam_info.header.stamp = ts_msg
        self._cam_info.header.frame_id = f'{self.ns[1:]}/camera_link'

        self.pub_rgb.publish(rgb_msg)
        self.pub_depth.publish(depth_msg)
        self.pub_info.publish(self._cam_info)

        if self.pub_clk:
            clk = Clock()
            clk.clock = ts_msg
            self.pub_clock.publish(clk)

    def _play(self):
        if not self.pairs:
            self.get_logger().error('No associated pairs found — check dataset path.')
            return

        # Skip the first start_skip_s seconds so RTAB-Map initialises on a
        # stable part of the sequence (avoids tracking failure on early
        # high-motion / low-feature frames).
        t0_data = self.pairs[0][0] + self.start_skip_s
        t0_wall = time.monotonic()

        for ts, f_rgb, f_depth in self.pairs:
            if ts < t0_data:
                continue
            elapsed_data = ts - t0_data
            elapsed_wall = time.monotonic() - t0_wall
            sleep = elapsed_data / self.rate - elapsed_wall
            if sleep > 0:
                time.sleep(sleep)
            self._publish_frame(ts, f_rgb, f_depth)

        self.get_logger().info('TUM player: finished playback.')


def main(args=None):
    rclpy.init(args=args)
    node = TUMPlayer()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
