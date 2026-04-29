#!/usr/bin/env python3
"""Decompresses + rectifies S3E stereo images for one robot.

S3E bags publish `/{Alpha,Bob,Carol}/{left,right}_camera/compressed`
(sensor_msgs/CompressedImage, 1224x1024, ~10 Hz). RTAB-Map stereo odometry
expects rectified `Image` pairs with matching `CameraInfo` (D=0, R=I).

This node takes one robot at a time: it reads the ORB-SLAM3-style
calibration YAML shipped with S3E (LEFT/RIGHT K, D, R, P), precomputes
undistort+rectify remap tables via `cv2.initUndistortRectifyMap`, then on
every compressed frame decodes, remaps, and republishes the rectified
image plus a rectified CameraInfo under the robot's cslam namespace.
"""
import os
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge
import yaml


def _matrix(calib, key):
    return np.array(calib[key]['data'], dtype=np.float64).reshape(
        calib[key]['rows'], calib[key]['cols'])


class S3EStereoRectifier(Node):

    def __init__(self):
        super().__init__('s3e_stereo_rectifier')

        self.declare_parameter('calibration_file', '')
        self.declare_parameter('left_compressed_topic', '')
        self.declare_parameter('right_compressed_topic', '')
        self.declare_parameter('left_image_topic', '')
        self.declare_parameter('right_image_topic', '')
        self.declare_parameter('left_info_topic', '')
        self.declare_parameter('right_info_topic', '')
        self.declare_parameter('left_frame_id', 'left_camera')
        self.declare_parameter('right_frame_id', 'right_camera')

        calib_file = self.get_parameter('calibration_file').value
        if not calib_file or not os.path.isfile(calib_file):
            raise RuntimeError(
                f's3e_stereo_rectifier: calibration_file not found: {calib_file}')

        left_comp = self.get_parameter('left_compressed_topic').value
        right_comp = self.get_parameter('right_compressed_topic').value
        left_img = self.get_parameter('left_image_topic').value
        right_img = self.get_parameter('right_image_topic').value
        left_info = self.get_parameter('left_info_topic').value
        right_info = self.get_parameter('right_info_topic').value
        self.left_frame = self.get_parameter('left_frame_id').value
        self.right_frame = self.get_parameter('right_frame_id').value

        with open(calib_file, 'r') as f:
            # Skip the leading `%YAML:1.0` directive OpenCV writes — PyYAML
            # trips on it otherwise.
            text = f.read()
            if text.startswith('%YAML'):
                text = text.split('\n', 1)[1]
            calib = yaml.safe_load(text)

        self.width = int(calib['LEFT.width'])
        self.height = int(calib['LEFT.height'])
        left_K  = _matrix(calib, 'LEFT.K')
        left_D  = _matrix(calib, 'LEFT.D').flatten()
        right_K = _matrix(calib, 'RIGHT.K')
        right_D = _matrix(calib, 'RIGHT.D').flatten()

        # True baseline from Camera.bf / Camera.fx (= 0.360 m for all S3E robots)
        true_baseline = float(calib['Camera.bf']) / float(calib['Camera.fx'])

        # Recover the rotation and translation between cameras from the
        # extrinsic Tlc (left_camera → lidar) matrices if available,
        # otherwise fall back to the baseline-only model.
        # For S3E the RIGHT.R in the YAML is close to identity (per-camera
        # calibration, NOT a joint stereo rectification), so we use
        # RIGHT.R as the rotation between cameras and the true baseline
        # for translation.
        right_R_raw = _matrix(calib, 'RIGHT.R')
        R = right_R_raw  # rotation from left to right camera
        T = np.array([-true_baseline, 0.0, 0.0])  # translation: right camera is to the right

        img_size = (self.width, self.height)

        # Joint stereo rectification — produces common focal length,
        # horizontal epipolar lines, and correct P matrices.
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            left_K, left_D, right_K, right_D, img_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

        self.left_map_x, self.left_map_y = cv2.initUndistortRectifyMap(
            left_K, left_D, R1, P1, img_size, cv2.CV_32FC1)
        self.right_map_x, self.right_map_y = cv2.initUndistortRectifyMap(
            right_K, right_D, R2, P2, img_size, cv2.CV_32FC1)

        baseline = -P2[0, 3] / P2[0, 0]
        self.get_logger().info(
            f'Loaded {calib_file}: {self.width}x{self.height}, '
            f'baseline={baseline:.4f}m (true={true_baseline:.4f}m), '
            f'fx={P1[0,0]:.2f}')

        self.left_info_msg = self._camera_info(P1, self.left_frame)
        self.right_info_msg = self._camera_info(P2, self.right_frame)

        self.bridge = CvBridge()
        self._lock = threading.Lock()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        self.left_img_pub = self.create_publisher(Image, left_img, sensor_qos)
        self.right_img_pub = self.create_publisher(Image, right_img, sensor_qos)
        self.left_info_pub = self.create_publisher(CameraInfo, left_info, sensor_qos)
        self.right_info_pub = self.create_publisher(CameraInfo, right_info, sensor_qos)

        # Bag player publishes with default (reliable) QoS — match it.
        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)
        self.create_subscription(
            CompressedImage, left_comp,
            lambda m: self._on_image(m, is_left=True), sub_qos)
        self.create_subscription(
            CompressedImage, right_comp,
            lambda m: self._on_image(m, is_left=False), sub_qos)

        self.get_logger().info(
            f'Subscribed: {left_comp}, {right_comp} → publishing {left_img}, {right_img}')

    def _camera_info(self, P, frame_id):
        msg = CameraInfo()
        msg.header.frame_id = frame_id
        msg.width = self.width
        msg.height = self.height
        msg.distortion_model = 'plumb_bob'
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.k = P[:3, :3].flatten().tolist()
        msg.r = np.eye(3, dtype=np.float64).flatten().tolist()
        msg.p = P.flatten().tolist()
        return msg

    def _on_image(self, msg: CompressedImage, is_left: bool):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            self.get_logger().warn('Failed to decode compressed frame')
            return

        if is_left:
            rect = cv2.remap(img, self.left_map_x, self.left_map_y, cv2.INTER_LINEAR)
            out = self.bridge.cv2_to_imgmsg(rect, encoding='bgr8')
            out.header = msg.header
            out.header.frame_id = self.left_frame
            with self._lock:
                self.left_img_pub.publish(out)
                self.left_info_msg.header = out.header
                self.left_info_pub.publish(self.left_info_msg)
        else:
            rect = cv2.remap(img, self.right_map_x, self.right_map_y, cv2.INTER_LINEAR)
            out = self.bridge.cv2_to_imgmsg(rect, encoding='bgr8')
            out.header = msg.header
            out.header.frame_id = self.right_frame
            with self._lock:
                self.right_img_pub.publish(out)
                self.right_info_msg.header = out.header
                self.right_info_pub.publish(self.right_info_msg)


def main():
    rclpy.init()
    node = S3EStereoRectifier()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
