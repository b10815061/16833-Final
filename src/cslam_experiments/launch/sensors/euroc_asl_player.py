#!/usr/bin/env python3
"""Plays a EuRoC ASL-format dataset as ROS 2 topics.

Reads images from mav0/cam0/data/, mav0/cam1/data/ and IMU from
mav0/imu0/data.csv, publishes them with proper timestamps and
camera_info (including distortion and stereo rectification).

Usage as a ROS 2 node — parameters:
  dataset_path: path to the sequence folder (containing mav0/)
  namespace: robot namespace (e.g., /r0)
  rate: playback speed multiplier (default 1.0)
  publish_clock: whether to publish /clock (default true)
"""
import csv
import os
import time
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import ClockType
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from builtin_interfaces.msg import Time as TimeMsg
from sensor_msgs.msg import Image, CameraInfo, Imu
from rosgraph_msgs.msg import Clock
from cv_bridge import CvBridge


# ── EuRoC calibration ────────────────────────────────────────────
CAM0_K = np.array([[458.654, 0, 367.215],
                    [0, 457.296, 248.375],
                    [0, 0, 1]], dtype=np.float64)
CAM0_D = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05],
                   dtype=np.float64)

CAM1_K = np.array([[457.587, 0, 379.999],
                    [0, 456.134, 255.238],
                    [0, 0, 1]], dtype=np.float64)
CAM1_D = np.array([-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05],
                   dtype=np.float64)

IMG_W, IMG_H = 752, 480

# T_cam0_cam1 from EuRoC (cam1 expressed in cam0 frame)
# Derived from T_BS_cam0 and T_BS_cam1 in sensor.yaml
T_BS_CAM0 = np.array([
    [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
    [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
    [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
    [0.0, 0.0, 0.0, 1.0]
])

T_BS_CAM1 = np.array([
    [0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556],
    [0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024],
    [-0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038],
    [0.0, 0.0, 0.0, 1.0]
])


def ns_to_time_msg(ns):
    msg = TimeMsg()
    msg.sec = int(ns // 1_000_000_000)
    msg.nanosec = int(ns % 1_000_000_000)
    return msg


def compute_stereo_rectification():
    # stereoRectify expects R, T that map cam0 -> cam1: p1 = R * p0 + T
    # T_BS transforms sensor to body, so T_cam1_cam0 = inv(T_BS_CAM1) @ T_BS_CAM0
    T_cam1_cam0 = np.linalg.inv(T_BS_CAM1) @ T_BS_CAM0
    R = T_cam1_cam0[:3, :3]
    T = T_cam1_cam0[:3, 3]

    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        CAM0_K, CAM0_D, CAM1_K, CAM1_D, (IMG_W, IMG_H), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

    # Build undistortion+rectification maps (applied once per frame)
    map0x, map0y = cv2.initUndistortRectifyMap(
        CAM0_K, CAM0_D, R1, P1, (IMG_W, IMG_H), cv2.CV_32FC1)
    map1x, map1y = cv2.initUndistortRectifyMap(
        CAM1_K, CAM1_D, R2, P2, (IMG_W, IMG_H), cv2.CV_32FC1)

    return R1, R2, P1, P2, map0x, map0y, map1x, map1y


class EurocAslPlayer(Node):

    def __init__(self):
        super().__init__('euroc_asl_player')

        self.declare_parameter('dataset_path', '')
        self.declare_parameter('namespace', '/r0')
        self.declare_parameter('rate', 1.0)
        self.declare_parameter('publish_clock', True)
        self.declare_parameter('frame_prefix', '')

        dataset_path = self.get_parameter('dataset_path').value
        ns = self.get_parameter('namespace').value
        self.rate = self.get_parameter('rate').value
        self.publish_clock = self.get_parameter('publish_clock').value
        self.frame_prefix = self.get_parameter('frame_prefix').value

        mav0 = os.path.join(dataset_path, 'mav0')
        self.bridge = CvBridge()

        # Compute stereo rectification + undistortion maps
        R1, R2, P1, P2, map0x, map0y, map1x, map1y = compute_stereo_rectification()
        self.map0x, self.map0y = map0x, map0y
        self.map1x, self.map1y = map1x, map1y

        # Build CameraInfo messages for rectified images:
        # D = zeros (distortion already removed), K = P[:3,:3], R = identity, P = from stereoRectify
        fp = self.frame_prefix
        rect_K0 = P1[:3, :3]
        rect_K1 = P2[:3, :3]
        self.left_info = self._make_camera_info(rect_K0, np.zeros(4), np.eye(3), P1, fp + 'cam0')
        self.right_info = self._make_camera_info(rect_K1, np.zeros(4), np.eye(3), P2, fp + 'cam1')

        # Publishers — use Best Effort QoS to match RTAB-Map subscribers (qos=2)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)
        self.left_img_pub = self.create_publisher(
            Image, ns + '/stereo_camera/left/image_rect_color', sensor_qos)
        self.right_img_pub = self.create_publisher(
            Image, ns + '/stereo_camera/right/image_rect_color', sensor_qos)
        self.left_info_pub = self.create_publisher(
            CameraInfo, ns + '/stereo_camera/left/camera_info', sensor_qos)
        self.right_info_pub = self.create_publisher(
            CameraInfo, ns + '/stereo_camera/right/camera_info', sensor_qos)
        self.imu_pub = self.create_publisher(
            Imu, ns + '/imu/data', QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=50))
        if self.publish_clock:
            self.clock_pub = self.create_publisher(Clock, '/clock', 10)

        # Load data
        self.cam0_data = self._load_image_csv(
            os.path.join(mav0, 'cam0', 'data.csv'),
            os.path.join(mav0, 'cam0', 'data'))
        self.cam1_data = self._load_image_csv(
            os.path.join(mav0, 'cam1', 'data.csv'),
            os.path.join(mav0, 'cam1', 'data'))
        self.imu_data = self._load_imu_csv(
            os.path.join(mav0, 'imu0', 'data.csv'))

        self.get_logger().info(
            f'Loaded {len(self.cam0_data)} left imgs, '
            f'{len(self.cam1_data)} right imgs, '
            f'{len(self.imu_data)} IMU samples. Rate={self.rate}x')

        # Start playback in a background thread so the executor can flush messages
        self._playback_thread = threading.Thread(target=self._run_playback, daemon=True)
        self._playback_thread.start()

    def _make_camera_info(self, K, D, R, P, frame_id):
        msg = CameraInfo()
        msg.header.frame_id = frame_id
        msg.width = IMG_W
        msg.height = IMG_H
        msg.distortion_model = 'plumb_bob'
        msg.d = D.tolist() + [0.0]
        msg.k = K.flatten().tolist()
        msg.r = R.flatten().tolist()
        msg.p = P.flatten().tolist()
        return msg

    def _load_image_csv(self, csv_path, data_dir):
        entries = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0].startswith('#'):
                    continue
                ts_ns = int(row[0])
                img_path = os.path.join(data_dir, row[1].strip())
                entries.append((ts_ns, img_path))
        return entries

    def _load_imu_csv(self, csv_path):
        entries = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0].startswith('#'):
                    continue
                ts_ns = int(row[0])
                wx, wy, wz = float(row[1]), float(row[2]), float(row[3])
                ax, ay, az = float(row[4]), float(row[5]), float(row[6])
                entries.append((ts_ns, wx, wy, wz, ax, ay, az))
        return entries

    def _run_playback(self):
        # Merge all events into a single sorted timeline
        events = []
        for ts_ns, img_path in self.cam0_data:
            events.append((ts_ns, 'cam0', img_path))
        for ts_ns, img_path in self.cam1_data:
            events.append((ts_ns, 'cam1', img_path))
        for ts_ns, wx, wy, wz, ax, ay, az in self.imu_data:
            events.append((ts_ns, 'imu', (wx, wy, wz, ax, ay, az)))
        events.sort(key=lambda x: x[0])

        if not events:
            self.get_logger().error('No data to play')
            return

        t0_data = events[0][0]
        t0_wall = time.monotonic()

        for i, event in enumerate(events):
            if not rclpy.ok():
                break

            ts_ns = event[0]
            elapsed_data = (ts_ns - t0_data) / 1e9
            target_wall = t0_wall + elapsed_data / self.rate

            now = time.monotonic()
            if target_wall > now:
                time.sleep(target_wall - now)

            stamp = ns_to_time_msg(ts_ns)

            if self.publish_clock:
                clock_msg = Clock()
                clock_msg.clock = stamp
                self.clock_pub.publish(clock_msg)

            fp = self.frame_prefix
            if event[1] == 'cam0':
                self._publish_image(event[2], stamp, fp + 'cam0',
                                     self.left_img_pub, self.left_info_pub,
                                     self.left_info,
                                     mapx=self.map0x, mapy=self.map0y)
            elif event[1] == 'cam1':
                self._publish_image(event[2], stamp, fp + 'cam1',
                                     self.right_img_pub, self.right_info_pub,
                                     self.right_info,
                                     mapx=self.map1x, mapy=self.map1y)
            elif event[1] == 'imu':
                self._publish_imu(stamp, event[2])

            if (i + 1) % 5000 == 0:
                self.get_logger().info(
                    f'Published {i+1}/{len(events)} events '
                    f'({100*(i+1)/len(events):.0f}%)')

        self.get_logger().info('Playback complete.')

    def _publish_image(self, img_path, stamp, frame_id, img_pub, info_pub, info_msg,
                       mapx=None, mapy=None):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return
        # Apply rectification maps if provided
        if mapx is not None and mapy is not None:
            img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        img_msg = self.bridge.cv2_to_imgmsg(img, encoding='mono8')
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = frame_id
        img_pub.publish(img_msg)

        info_msg.header.stamp = stamp
        info_msg.header.frame_id = frame_id
        info_pub.publish(info_msg)

    def _publish_imu(self, stamp, data):
        wx, wy, wz, ax, ay, az = data
        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_prefix + 'imu0'
        msg.angular_velocity.x = wx
        msg.angular_velocity.y = wy
        msg.angular_velocity.z = wz
        msg.linear_acceleration.x = ax
        msg.linear_acceleration.y = ay
        msg.linear_acceleration.z = az
        self.imu_pub.publish(msg)


def main():
    rclpy.init()
    node = EurocAslPlayer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
