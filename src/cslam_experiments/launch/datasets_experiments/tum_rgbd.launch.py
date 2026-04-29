"""Launch Swarm-SLAM on TUM RGB-D dataset (single or multi-robot).

Usage (single):
  ros2 launch cslam_experiments tum_rgbd.launch.py \
      sequences:=rgbd_dataset_freiburg1_desk \
      dataset_path:=/home/ros/datasets/TUM \
      config_file:=tum_baseline.yaml

Usage (multi-agent):
  ros2 launch cslam_experiments tum_rgbd.launch.py \
      max_nb_robots:=2 \
      sequences:=rgbd_dataset_freiburg1_desk,rgbd_dataset_freiburg1_desk2 \
      dataset_path:=/home/ros/datasets/TUM \
      config_file:=tum_baseline.yaml
"""
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
    pkg      = get_package_share_directory('cslam_experiments')
    cfg_path = os.path.join(pkg, 'config/')

    config_file   = LaunchConfiguration('config_file').perform(context)
    max_nb_robots = int(LaunchConfiguration('max_nb_robots').perform(context))
    dataset_path  = LaunchConfiguration('dataset_path').perform(context)
    robot_delay_s  = float(LaunchConfiguration('robot_delay_s').perform(context))
    launch_delay_s = float(LaunchConfiguration('launch_delay_s').perform(context))
    start_skip_s   = float(LaunchConfiguration('start_skip_s').perform(context))
    rate           = float(LaunchConfiguration('rate').perform(context))
    sequences      = LaunchConfiguration('sequences').perform(context).split(',')

    robot_delay_s  /= rate
    launch_delay_s /= rate

    # Read first timestamp of each sequence so all robots share the same clock base.
    # Robot 0 drives the /clock; robot i's ts_offset_s shifts its timestamps so that
    # its first frame arrives at the current sim clock when its player starts.
    #
    # Robot 0 player starts at sim time T0_robot0.
    # Robot i player starts robot_delay_s*i (real seconds) later.
    # During that delay, sim clock advances robot_delay_s*rate*i seconds.
    # So robot i's first frame must have timestamp T0_robot0 + robot_delay_s*rate*i.
    # ts_offset_i = (T0_robot0 + robot_delay_s*rate*i) - T0_seq_i
    #             = (T0_robot0 - T0_seq_i) + robot_delay_s * rate * i
    def _first_ts(seq_path):
        rgb_txt = os.path.join(seq_path, 'rgb.txt')
        with open(rgb_txt) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    return float(line.split()[0])
        return 0.0

    seq_paths = [os.path.join(dataset_path, s.strip()) for s in sequences]
    t0_robot0 = _first_ts(seq_paths[0]) if seq_paths else 0.0
    ts_offsets = [(t0_robot0 - _first_ts(seq_paths[i]) + robot_delay_s * rate * i)
                  if i < len(seq_paths) else 0.0
                  for i in range(max_nb_robots)]

    cslam_procs = []
    player_procs = []
    odom_procs  = []

    for i in range(max_nb_robots):
        prefix = f'r{i}'

        # CSLAM core (RGB-D)
        cslam_procs.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg, 'launch', 'cslam', 'cslam_rgbd.launch.py')),
            launch_arguments={
                'config_path': cfg_path,
                'config_file': config_file,
                'robot_id':    str(i),
                'namespace':   f'/{prefix}',
                'max_nb_robots': str(max_nb_robots),
                'enable_simulated_rendezvous': 'false',
                'rendezvous_schedule_file': '',
                'sensor_base_frame_id': f'{prefix}/camera_link',
            }.items(),
        ))

        # TUM player
        seq_path = seq_paths[i] if i < len(seq_paths) else seq_paths[0]
        player_procs.append(Node(
            package='cslam_experiments',
            executable='tum_player.py',
            name=f'tum_player_{prefix}',
            parameters=[{
                'dataset_path':  seq_path,
                'namespace':     f'/{prefix}',
                'rate':          rate,
                'publish_clock': (i == 0),
                'ts_offset_s':   ts_offsets[i],
                'start_skip_s':  start_skip_s,
            }],
            output='screen',
        ))

        # RTAB-Map RGB-D odometry
        odom_procs.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg, 'launch', 'odometry',
                             'rtabmap_rgbd_odometry.launch.py')),
            launch_arguments={
                'namespace':    f'/{prefix}',
                'frame_id':     f'{prefix}/base_link',
                'vo_frame_id':  f'{prefix}/odom',
            }.items(),
        ))

    # TF chain for each robot
    tf_nodes = []
    for i in range(max_nb_robots):
        p = f'r{i}'
        tf_nodes += [
            Node(package='tf2_ros', executable='static_transform_publisher',
                 name=f'tf_map_odom_{p}',
                 arguments=['--frame-id', f'robot{i}_map',
                            '--child-frame-id', f'{p}/odom',
                            '--x','0','--y','0','--z','0',
                            '--qx','0','--qy','0','--qz','0','--qw','1']),
            Node(package='tf2_ros', executable='static_transform_publisher',
                 name=f'tf_base_cam_{p}',
                 arguments=['--frame-id', f'{p}/base_link',
                            '--child-frame-id', f'{p}/camera_link',
                            '--x','0','--y','0','--z','0',
                            '--qx','0','--qy','0','--qz','0','--qw','1']),
        ]

    schedule = []
    for tf in tf_nodes:
        schedule += [PushLaunchConfigurations(), tf, PopLaunchConfigurations()]

    for i in range(max_nb_robots):
        delay = robot_delay_s * i
        schedule += [
            PushLaunchConfigurations(),
            TimerAction(period=delay, actions=[cslam_procs[i]]),
            PopLaunchConfigurations(),
            PushLaunchConfigurations(),
            TimerAction(period=delay, actions=[odom_procs[i]]),
            PopLaunchConfigurations(),
        ]

    for i in range(max_nb_robots):
        schedule += [
            PushLaunchConfigurations(),
            TimerAction(period=robot_delay_s * i + launch_delay_s,
                        actions=[player_procs[i]]),
            PopLaunchConfigurations(),
        ]

    return schedule


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('max_nb_robots',    default_value='1'),
        DeclareLaunchArgument('sequences',
                              default_value='rgbd_dataset_freiburg1_desk'),
        DeclareLaunchArgument('dataset_path',
                              default_value='/home/ros/datasets/TUM'),
        DeclareLaunchArgument('robot_delay_s',    default_value='0'),
        DeclareLaunchArgument('launch_delay_s',   default_value='10'),
        DeclareLaunchArgument('start_skip_s',     default_value='3.0'),
        DeclareLaunchArgument('config_file',      default_value='tum_baseline.yaml'),
        DeclareLaunchArgument('rate',             default_value='1.0'),
        OpaqueFunction(function=launch_setup),
    ])
