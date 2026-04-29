from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def launch_setup(context, *args, **kwargs):
    ns = LaunchConfiguration('namespace').perform(context)
    # Allow explicit frame_id/vo_frame_id overrides; fall back to legacy default
    frame_id = LaunchConfiguration('frame_id').perform(context) or ns[1:] + '_link'
    vo_frame_id = LaunchConfiguration('vo_frame_id').perform(context) or ns[1:] + '/odom'

    parameters=[{
          'frame_id': frame_id,
          'odom_frame_id': vo_frame_id,
          'subscribe_depth':True,
          'approx_sync':True,
          'Vis/MaxFeatures': '2000',
          'Vis/MinInliers': '10',
          'RGBD/DepthMin': '0.1',
          'RGBD/DepthMax': '4.0',
          'Odom/ResetCountdown': '5',
          }]

    remappings=[
          ('rgb/image', ns + '/color/image_raw'),
          ('rgb/camera_info', ns + '/color/camera_info'),
          ('depth/image', ns + '/aligned_depth_to_color/image_raw'),
          ('odom', ns + '/odom'),]

    return [
        Node(
            package='rtabmap_odom', executable='rgbd_odometry', output='screen', name='rgbd_odometry',
            parameters=parameters,
            remappings=remappings,
            ),
    ]

def generate_launch_description():

    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='/r0',
                              description='Robot namespace'),
        DeclareLaunchArgument('frame_id', default_value='',
                              description='Robot base frame (empty = use namespace-derived default)'),
        DeclareLaunchArgument('vo_frame_id', default_value='',
                              description='Odometry output frame (empty = use namespace-derived default)'),
        OpaqueFunction(function=launch_setup)
    ])
