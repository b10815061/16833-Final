#!/usr/bin/env python3
"""Publishes CameraInfo messages for EuRoC MAV stereo cameras.

EuRoC bags don't include camera_info topics and images are unrectified.
We publish full calibration (with distortion) so RTAB-Map can rectify internally.
Stereo rectification matrices (R1, R2, P1, P2) are computed once at startup.
"""
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo


# EuRoC cam0 (left) intrinsics from sensor.yaml
CAM0_FX, CAM0_FY = 458.654, 457.296
CAM0_CX, CAM0_CY = 367.215, 248.375
CAM0_DIST = [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]

# EuRoC cam1 (right) intrinsics from sensor.yaml
CAM1_FX, CAM1_FY = 457.587, 456.134
CAM1_CX, CAM1_CY = 379.999, 255.238
CAM1_DIST = [-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05]

IMG_W, IMG_H = 752, 480

# Extrinsics: T_cam1_cam0 (cam1 w.r.t. cam0) from EuRoC calibration
# Translation in meters
T_CAM1_CAM0 = np.array([
    [ 0.9999758,   0.00580413, -0.00384223, -0.11007381],
    [-0.00581534,  0.99998049, -0.00249853,  0.00039912],
    [ 0.00382767,  0.00252085,  0.99998947, -0.00054831],
    [ 0.0,         0.0,         0.0,          1.0       ]
])


def compute_stereo_rectification():
    """Compute stereo rectification using OpenCV."""
    K0 = np.array([[CAM0_FX, 0, CAM0_CX],
                    [0, CAM0_FY, CAM0_CY],
                    [0, 0, 1]], dtype=np.float64)
    D0 = np.array(CAM0_DIST, dtype=np.float64)

    K1 = np.array([[CAM1_FX, 0, CAM1_CX],
                    [0, CAM1_FY, CAM1_CY],
                    [0, 0, 1]], dtype=np.float64)
    D1 = np.array(CAM1_DIST, dtype=np.float64)

    # T_CAM1_CAM0 transforms points from cam0 to cam1 frame
    # stereoRectify expects R, T such that: p_cam1 = R * p_cam0 + T
    R = T_CAM1_CAM0[:3, :3]
    T = T_CAM1_CAM0[:3, 3]

    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        K0, D0, K1, D1, (IMG_W, IMG_H), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

    return K0, D0, K1, D1, R1, R2, P1, P2


class EurocCameraInfoPublisher(Node):

    def __init__(self):
        super().__init__('euroc_camera_info_publisher')

        self.declare_parameter('namespace', '/r0')
        ns = self.get_parameter('namespace').value

        # Compute stereo rectification once
        K0, D0, K1, D1, R1, R2, P1, P2 = compute_stereo_rectification()

        # Publishers
        self.left_info_pub = self.create_publisher(
            CameraInfo, ns + '/stereo_camera/left/camera_info', 10)
        self.right_info_pub = self.create_publisher(
            CameraInfo, ns + '/stereo_camera/right/camera_info', 10)

        # Subscribe to images to sync timestamps
        self.create_subscription(
            Image, ns + '/stereo_camera/left/image_rect_color',
            self.left_image_cb, 10)
        self.create_subscription(
            Image, ns + '/stereo_camera/right/image_rect_color',
            self.right_image_cb, 10)

        # cam0 (left) CameraInfo
        self.left_info = CameraInfo()
        self.left_info.width = IMG_W
        self.left_info.height = IMG_H
        self.left_info.distortion_model = 'plumb_bob'
        self.left_info.d = D0.tolist() + [0.0]  # 5 coefficients for plumb_bob
        self.left_info.k = K0.flatten().tolist()
        self.left_info.r = R1.flatten().tolist()
        self.left_info.p = P1.flatten().tolist()

        # cam1 (right) CameraInfo
        self.right_info = CameraInfo()
        self.right_info.width = IMG_W
        self.right_info.height = IMG_H
        self.right_info.distortion_model = 'plumb_bob'
        self.right_info.d = D1.tolist() + [0.0]  # 5 coefficients for plumb_bob
        self.right_info.k = K1.flatten().tolist()
        self.right_info.r = R2.flatten().tolist()
        self.right_info.p = P2.flatten().tolist()

        self.get_logger().info(
            f'EuRoC stereo calibration ready. '
            f'P1[0,3]={P1[0,3]:.2f}, P2[0,3]={P2[0,3]:.2f}, '
            f'baseline={abs(P2[0,3]/P2[0,0]):.4f}m')

    def left_image_cb(self, msg):
        self.left_info.header = msg.header
        self.left_info_pub.publish(self.left_info)

    def right_image_cb(self, msg):
        self.right_info.header = msg.header
        self.right_info_pub.publish(self.right_info)


def main():
    rclpy.init()
    node = EurocCameraInfoPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
