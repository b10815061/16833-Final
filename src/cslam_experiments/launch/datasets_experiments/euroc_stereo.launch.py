import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (TimerAction, OpaqueFunction, PushLaunchConfigurations,
                            PopLaunchConfigurations, DeclareLaunchArgument,
                            IncludeLaunchDescription)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def launch_setup(context, *args, **kwargs):
    config_path = os.path.join(
        get_package_share_directory("cslam_experiments"), "config/")
    config_file = LaunchConfiguration('config_file').perform(context)

    max_nb_robots = int(LaunchConfiguration('max_nb_robots').perform(context))
    dataset_path = LaunchConfiguration('dataset_path').perform(context)
    robot_delay_s = float(LaunchConfiguration('robot_delay_s').perform(context))
    launch_delay_s = float(LaunchConfiguration('launch_delay_s').perform(context))
    rate = float(LaunchConfiguration('rate').perform(context))

    # Sequences per robot — comma-separated, e.g., "MH_01_easy,MH_02_easy"
    sequences = LaunchConfiguration('sequences').perform(context).split(',')

    robot_delay_s = robot_delay_s / rate
    launch_delay_s = launch_delay_s / rate

    cslam_processes = []
    player_processes = []
    odom_processes = []

    for i in range(max_nb_robots):
        # CSLAM core
        proc = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory("cslam_experiments"),
                             "launch", "cslam", "cslam_stereo.launch.py")),
            launch_arguments={
                "config_path": config_path,
                "config_file": config_file,
                "robot_id": str(i),
                "namespace": "/r" + str(i),
                "max_nb_robots": str(max_nb_robots),
                "sensor_base_frame_id": "r" + str(i) + "/camera_link",
                "enable_simulated_rendezvous": LaunchConfiguration('enable_simulated_rendezvous'),
                "rendezvous_schedule_file": os.path.join(
                    config_path, "rendezvous",
                    LaunchConfiguration('rendezvous_config').perform(context)),
            }.items(),
        )
        cslam_processes.append(proc)

        # ASL dataset player (replaces rosbag)
        seq_path = os.path.join(dataset_path, sequences[i].strip())
        player_proc = Node(
            package='cslam_experiments',
            executable='euroc_asl_player.py',
            name='euroc_player_r' + str(i),
            parameters=[{
                'dataset_path': seq_path,
                'namespace': '/r' + str(i),
                'rate': rate,
                'publish_clock': (i == 0),  # Only first robot publishes clock
                'frame_prefix': 'r' + str(i) + '/',
            }],
            output='screen',
        )
        player_processes.append(player_proc)

        # Stereo odometry via RTAB-Map
        prefix = "r" + str(i)
        odom_proc = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory("cslam_experiments"),
                             "launch", "odometry",
                             "rtabmap_kitti_stereo_odometry.launch.py")),
            launch_arguments={
                "namespace": "/" + prefix,
                "log_level": "error",
                "robot_id": str(i),
                "approx_sync": "true",
                "frame_id": prefix + "/base_link",
                "vo_frame_id": prefix + "/odom",
            }.items(),
        )
        odom_processes.append(odom_proc)

    # Per-robot TF chains: robot{i}_map -> r{i}/odom -> r{i}/base_link -> r{i}/camera_link -> r{i}/cam0 -> r{i}/cam1
    tf_nodes = []
    for i in range(max_nb_robots):
        prefix = f"r{i}"
        tf_nodes.append(Node(
            package="tf2_ros", executable="static_transform_publisher",
            name=f"tf_map_odom_r{i}",
            arguments=["--frame-id", f"robot{i}_map", "--child-frame-id", f"{prefix}/odom",
                        "--x", "0", "--y", "0", "--z", "0",
                        "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1"],
        ))
        tf_nodes.append(Node(
            package="tf2_ros", executable="static_transform_publisher",
            name=f"tf_base_camera_r{i}",
            arguments=["--frame-id", f"{prefix}/base_link", "--child-frame-id", f"{prefix}/camera_link",
                        "--x", "0", "--y", "0", "--z", "0",
                        "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1"],
        ))
        tf_nodes.append(Node(
            package="tf2_ros", executable="static_transform_publisher",
            name=f"tf_camera_cam0_r{i}",
            arguments=["--frame-id", f"{prefix}/camera_link", "--child-frame-id", f"{prefix}/cam0",
                        "--x", "0", "--y", "0", "--z", "0",
                        "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1"],
        ))
        tf_nodes.append(Node(
            package="tf2_ros", executable="static_transform_publisher",
            name=f"tf_cam0_cam1_r{i}",
            arguments=["--frame-id", f"{prefix}/cam0", "--child-frame-id", f"{prefix}/cam1",
                        "--x", "0.110074", "--y", "0", "--z", "0",
                        "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1"],
        ))

    # Launch schedule
    schedule = []

    # TF first
    for tf_node in tf_nodes:
        schedule.append(PushLaunchConfigurations())
        schedule.append(tf_node)
        schedule.append(PopLaunchConfigurations())

    # CSLAM + odometry nodes
    for i in range(max_nb_robots):
        schedule.append(PushLaunchConfigurations())
        schedule.append(
            TimerAction(period=robot_delay_s * i,
                        actions=[cslam_processes[i]]))
        schedule.append(PopLaunchConfigurations())
        schedule.append(PushLaunchConfigurations())
        schedule.append(
            TimerAction(period=robot_delay_s * i,
                        actions=[odom_processes[i]]))
        schedule.append(PopLaunchConfigurations())

    # Dataset players (delayed to let nodes initialize)
    for i in range(max_nb_robots):
        schedule.append(PushLaunchConfigurations())
        schedule.append(
            TimerAction(period=robot_delay_s * i + launch_delay_s,
                        actions=[player_processes[i]]))
        schedule.append(PopLaunchConfigurations())

    return schedule


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('max_nb_robots', default_value='1'),
        DeclareLaunchArgument('sequences', default_value='MH_01_easy',
                              description='Comma-separated EuRoC sequence folder names'),
        DeclareLaunchArgument('dataset_path', default_value='/root/datasets/euroc',
                              description='Path containing sequence folders (each with mav0/)'),
        DeclareLaunchArgument('robot_delay_s', default_value='0',
                              description='Delay between launching each robot'),
        DeclareLaunchArgument('launch_delay_s', default_value='10',
                              description='Delay between launching nodes and data playback'),
        DeclareLaunchArgument('config_file', default_value='euroc_stereo.yaml'),
        DeclareLaunchArgument('rate', default_value='1.0'),
        DeclareLaunchArgument('enable_simulated_rendezvous', default_value='false'),
        DeclareLaunchArgument('rendezvous_config', default_value='kitti00_5robots.config'),
        OpaqueFunction(function=launch_setup),
    ])
