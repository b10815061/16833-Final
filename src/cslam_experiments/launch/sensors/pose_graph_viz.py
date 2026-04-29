#!/usr/bin/env python3
"""Subscribes to /cslam/viz/pose_graph (custom PoseGraph msg) and publishes
a visualization_msgs/MarkerArray that RViz2 can display.

- Nodes: colored spheres at each pose (color per robot_id)
- Edges: lines connecting poses (green = intra-robot, red = inter-robot)

Usage:
  ros2 run cslam_experiments pose_graph_viz.py
"""
import rclpy
from rclpy.node import Node
from cslam_common_interfaces.msg import PoseGraph
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point


# Per-robot colors (up to 8 robots)
ROBOT_COLORS = [
    ColorRGBA(r=0.2, g=0.6, b=1.0, a=1.0),   # blue
    ColorRGBA(r=1.0, g=0.4, b=0.1, a=1.0),   # orange
    ColorRGBA(r=0.2, g=0.8, b=0.2, a=1.0),   # green
    ColorRGBA(r=0.8, g=0.2, b=0.8, a=1.0),   # purple
    ColorRGBA(r=1.0, g=0.8, b=0.0, a=1.0),   # yellow
    ColorRGBA(r=0.0, g=0.8, b=0.8, a=1.0),   # cyan
    ColorRGBA(r=1.0, g=0.2, b=0.2, a=1.0),   # red
    ColorRGBA(r=0.6, g=0.6, b=0.6, a=1.0),   # gray
]

INTER_ROBOT_EDGE_COLOR = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
INTRA_ROBOT_EDGE_COLOR = ColorRGBA(r=0.0, g=0.8, b=0.0, a=0.5)


class PoseGraphViz(Node):

    def __init__(self):
        super().__init__('pose_graph_viz')
        self.sub = self.create_subscription(
            PoseGraph, '/cslam/viz/pose_graph', self.callback, 10)
        self.pub = self.create_publisher(
            MarkerArray, '/cslam/viz/pose_graph_markers', 10)
        self.get_logger().info('Pose graph visualizer ready')

    def callback(self, msg: PoseGraph):
        markers = MarkerArray()
        frame_id = f'robot{msg.origin_robot_id}_map'
        stamp = self.get_clock().now().to_msg()

        # Build a lookup: (robot_id, keyframe_id) -> position
        pose_lookup = {}
        for v in msg.values:
            rid = v.key.robot_id
            kid = v.key.keyframe_id
            pose_lookup[(rid, kid)] = v.pose.position

        # --- Node markers (one sphere list per robot) ---
        robot_points = {}
        for v in msg.values:
            rid = v.key.robot_id
            if rid not in robot_points:
                robot_points[rid] = []
            p = Point(x=v.pose.position.x,
                      y=v.pose.position.y,
                      z=v.pose.position.z)
            robot_points[rid].append(p)

        marker_id = 0
        for rid, points in robot_points.items():
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = stamp
            m.ns = f'nodes_r{rid}'
            m.id = marker_id
            m.type = Marker.SPHERE_LIST
            m.action = Marker.ADD
            m.scale.x = 0.15
            m.scale.y = 0.15
            m.scale.z = 0.15
            m.color = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
            m.points = points
            m.pose.orientation.w = 1.0
            markers.markers.append(m)
            marker_id += 1

        # --- Edge markers ---
        intra_points = []
        inter_points = []
        for e in msg.edges:
            key_from = (e.key_from.robot_id, e.key_from.keyframe_id)
            key_to = (e.key_to.robot_id, e.key_to.keyframe_id)
            if key_from not in pose_lookup or key_to not in pose_lookup:
                continue
            p1 = pose_lookup[key_from]
            p2 = pose_lookup[key_to]
            pt1 = Point(x=p1.x, y=p1.y, z=p1.z)
            pt2 = Point(x=p2.x, y=p2.y, z=p2.z)
            if e.key_from.robot_id == e.key_to.robot_id:
                intra_points.extend([pt1, pt2])
            else:
                inter_points.extend([pt1, pt2])

        if intra_points:
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = stamp
            m.ns = 'edges_intra'
            m.id = marker_id
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.scale.x = 0.03
            m.color = INTRA_ROBOT_EDGE_COLOR
            m.points = intra_points
            m.pose.orientation.w = 1.0
            markers.markers.append(m)
            marker_id += 1

        if inter_points:
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = stamp
            m.ns = 'edges_inter'
            m.id = marker_id
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.scale.x = 0.05
            m.color = INTER_ROBOT_EDGE_COLOR
            m.points = inter_points
            m.pose.orientation.w = 1.0
            markers.markers.append(m)
            marker_id += 1

        # Delete old markers that no longer exist
        clear = Marker()
        clear.header.frame_id = frame_id
        clear.header.stamp = stamp
        clear.ns = ''
        clear.id = 0
        clear.action = Marker.DELETEALL
        # Prepend DELETEALL so old markers are cleared before new ones drawn
        markers.markers.insert(0, clear)

        self.pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = PoseGraphViz()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
