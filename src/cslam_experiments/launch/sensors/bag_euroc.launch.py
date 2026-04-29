import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    ns = LaunchConfiguration('namespace').perform(context)

    # Camera info publisher (EuRoC bags don't include camera_info)
    camera_info_node = Node(
        package='cslam_experiments',
        executable='euroc_camera_info_publisher.py',
        name='euroc_camera_info_publisher',
        parameters=[{'namespace': ns}],
        output='screen',
    )

    bag_play = TimerAction(
        period=LaunchConfiguration('bag_start_delay'),
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'play',
                    LaunchConfiguration('bag_file').perform(context),
                    '-r', LaunchConfiguration('rate'),
                    '--clock',
                    '--remap',
                    # EuRoC stereo images (cam0=left, cam1=right)
                    '/cam0/image_raw:=' + ns + '/stereo_camera/left/image_rect_color',
                    '/cam1/image_raw:=' + ns + '/stereo_camera/right/image_rect_color',
                    # IMU
                    '/imu0:=' + ns + '/imu/data',
                ],
                name='bag',
                output='screen',
            )
        ])

    return [
        DeclareLaunchArgument('bag_file', default_value='', description=''),
        DeclareLaunchArgument('namespace', default_value='/r0', description=''),
        DeclareLaunchArgument('rate', default_value='1.0', description=''),
        DeclareLaunchArgument('bag_start_delay', default_value='5.0', description=''),
        camera_info_node,
        bag_play,
    ]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=launch_setup)])
