cmake_minimum_required(VERSION 3.5)
project(cslam_experiments)

install(DIRECTORY
   config
   launch
   DESTINATION share/${PROJECT_NAME}/
)

find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)

find_package(cslam REQUIRED)
find_package(cslam_common_interfaces REQUIRED)
find_package(cslam_common_interfaces REQUIRED)

find_package(launch_testing_ament_cmake REQUIRED)

find_package(ament_cmake_python REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
cslam_experiments/publish_stereo_calibration_s3e.py
launch/sensors/euroc_camera_info_publisher.py
launch/sensors/euroc_asl_player.py
launch/sensors/tum_player.py
launch/sensors/pose_graph_viz.py
DESTINATION lib/${PROJECT_NAME})

ament_package()
