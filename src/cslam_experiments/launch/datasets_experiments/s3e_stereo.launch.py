import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (TimerAction, OpaqueFunction, PushLaunchConfigurations,
                            PopLaunchConfigurations, DeclareLaunchArgument,
                            IncludeLaunchDescription, ExecuteProcess, Shutdown,
                            RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


ROBOT_NAMES = ['Alpha', 'Bob', 'Carol']
ROBOT_CALIBS = ['alpha.yaml', 'bob.yaml', 'carol.yaml']


def launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory('cslam_experiments')
    config_path = os.path.join(pkg_share, 'config/')
    config_file = LaunchConfiguration('config_file').perform(context)

    max_nb_robots = int(LaunchConfiguration('max_nb_robots').perform(context))
    dataset_path = LaunchConfiguration('dataset_path').perform(context)
    sequence = LaunchConfiguration('sequence').perform(context)
    robot_delay_s = float(LaunchConfiguration('robot_delay_s').perform(context))
    launch_delay_s = float(LaunchConfiguration('launch_delay_s').perform(context))
    rate = float(LaunchConfiguration('rate').perform(context))

    robot_delay_s = robot_delay_s / rate
    launch_delay_s = launch_delay_s / rate

    cslam_processes = []
    rectifier_processes = []
    odom_processes = []

    for i in range(max_nb_robots):
        prefix = f'r{i}'
        ns = '/' + prefix
        robot_name = ROBOT_NAMES[i]
        calib_file = os.path.join(pkg_share, 'config', 's3e', ROBOT_CALIBS[i])

        # CSLAM core
        cslam_processes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_share, 'launch', 'cslam', 'cslam_stereo.launch.py')),
            launch_arguments={
                'config_path': config_path,
                'config_file': config_file,
                'robot_id': str(i),
                'namespace': ns,
                'max_nb_robots': str(max_nb_robots),
                'sensor_base_frame_id': 'left_camera',
                'enable_simulated_rendezvous': LaunchConfiguration('enable_simulated_rendezvous'),
                'rendezvous_schedule_file': os.path.join(
                    config_path, 'rendezvous',
                    LaunchConfiguration('rendezvous_config').perform(context)),
            }.items(),
        ))

        # S3E compressed-stereo decompression + rectification
        rectifier_processes.append(Node(
            package='cslam_experiments',
            executable='s3e_stereo_rectifier.py',
            name=f's3e_rectifier_{prefix}',
            parameters=[{
                'calibration_file': calib_file,
                'left_compressed_topic':  f'/{robot_name}/left_camera/compressed',
                'right_compressed_topic': f'/{robot_name}/right_camera/compressed',
                'left_image_topic':  f'{ns}/stereo_camera/left/image_rect_color',
                'right_image_topic': f'{ns}/stereo_camera/right/image_rect_color',
                'left_info_topic':   f'{ns}/stereo_camera/left/camera_info',
                'right_info_topic':  f'{ns}/stereo_camera/right/camera_info',
                'left_frame_id':  'left_camera',
                'right_frame_id': 'right_camera',
            }],
            output='screen',
        ))

        # Lidar ICP odometry (much more accurate than stereo on outdoor S3E)
        odom_processes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_share, 'launch', 'odometry',
                             'rtabmap_s3e_lidar_odometry.launch.py')),
            launch_arguments={
                'namespace': ns,
                'log_level': 'error',
                'robot_id': str(i),
                'use_sim_time': 'true',
                'wait_imu_to_init': 'false',
                'imu_topic': '/null_imu',
            }.items(),
        ))

    # Per-robot TF chains
    # Lidar odom (default frame_id=velodyne, vo_frame_id=odom) publishes
    # odom→velodyne. The bag pointclouds have frame_id="velodyne" (unprefixed).
    # Use unprefixed frames to match upstream — same as bag_s3e.launch.py.
    tf_nodes = []
    baseline = 0.360
    for i in range(max_nb_robots):
        prefix = f'r{i}'
        tf_nodes.append(Node(
            package='tf2_ros', executable='static_transform_publisher',
            name=f'tf_map_odom_{prefix}',
            arguments=['--frame-id', f'robot{i}_map', '--child-frame-id', 'odom',
                       '--x', '0', '--y', '0', '--z', '0',
                       '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1'],
        ))
    # Shared static TFs (unprefixed, matching upstream bag_s3e.launch.py)
    # velodyne → left_camera (from S3E Tlc calibration)
    tf_nodes.append(Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='tf_velodyne_left_camera',
        arguments=['0.13', '0.20', '-0.13', '0.5', '-0.5', '0.5', '-0.5',
                   'left_camera', 'velodyne'],
    ))
    tf_nodes.append(Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='tf_velodyne_imu',
        arguments=['-0.051', '0.0195', '-0.094', '0', '0', '0', '1',
                   'velodyne', 'imu_link'],
    ))

    # Bag playback — single bag contains all 3 robots' topics. Remap IMU
    # topics into the r{i} namespaces; camera topics are left under the
    # /{Alpha,Bob,Carol}/ namespace so the rectifier can subscribe to them.
    bag_file = os.path.join(dataset_path, sequence, sequence + '.db3')
    if not os.path.isfile(bag_file):
        # rosbag2 Humble accepts either the directory or the .db3 path
        bag_file = os.path.join(dataset_path, sequence)

    bag_cmd = ['ros2', 'bag', 'play', bag_file, '--clock', '-r', str(rate),
               '--remap']
    for i in range(max_nb_robots):
        bag_cmd += [f'/{ROBOT_NAMES[i]}/imu/data:=/r{i}/imu/data',
                    f'/{ROBOT_NAMES[i]}/velodyne_points:=/r{i}/pointcloud']

    bag_proc = ExecuteProcess(cmd=bag_cmd, name='s3e_bag_play', output='screen')

    # Schedule
    schedule = []
    for tf_node in tf_nodes:
        schedule.append(PushLaunchConfigurations())
        schedule.append(tf_node)
        schedule.append(PopLaunchConfigurations())

    for i in range(max_nb_robots):
        schedule.append(PushLaunchConfigurations())
        schedule.append(TimerAction(period=robot_delay_s * i,
                                    actions=[cslam_processes[i]]))
        schedule.append(PopLaunchConfigurations())
        schedule.append(PushLaunchConfigurations())
        schedule.append(TimerAction(period=robot_delay_s * i,
                                    actions=[odom_processes[i]]))
        schedule.append(PopLaunchConfigurations())
        schedule.append(PushLaunchConfigurations())
        schedule.append(TimerAction(period=robot_delay_s * i,
                                    actions=[rectifier_processes[i]]))
        schedule.append(PopLaunchConfigurations())

    schedule.append(PushLaunchConfigurations())
    schedule.append(TimerAction(period=launch_delay_s, actions=[bag_proc]))
    schedule.append(PopLaunchConfigurations())

    shutdown_delay = float(LaunchConfiguration('shutdown_delay_s').perform(context))
    schedule.append(RegisterEventHandler(OnProcessExit(
        target_action=bag_proc,
        on_exit=[TimerAction(period=shutdown_delay,
                             actions=[Shutdown(reason='Bag playback finished')])],
    )))

    return schedule


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('max_nb_robots', default_value='1'),
        DeclareLaunchArgument('sequence', default_value='S3E_Playground_1',
                              description='S3E sequence folder name (e.g. S3E_Playground_1)'),
        DeclareLaunchArgument('dataset_path',
                              default_value='/root/datasets/S3E/S3Ev1',
                              description='Directory containing sequence folders'),
        DeclareLaunchArgument('robot_delay_s', default_value='0'),
        DeclareLaunchArgument('launch_delay_s', default_value='10',
                              description='Delay between launching nodes and bag playback'),
        DeclareLaunchArgument('config_file', default_value='s3e_stereo.yaml'),
        DeclareLaunchArgument('rate', default_value='1.0'),
        DeclareLaunchArgument('enable_simulated_rendezvous', default_value='false'),
        DeclareLaunchArgument('shutdown_delay_s', default_value='30',
                              description='Seconds to wait after bag ends before shutting down '
                                          '(allows final PGO snapshot to save)'),
        DeclareLaunchArgument('rendezvous_config',
                              default_value='s3e_playground.config'),
        OpaqueFunction(function=launch_setup),
    ])
