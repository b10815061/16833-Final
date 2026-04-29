FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

ARG USER_UID=1000
ARG USER_GID=1000

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    python3-pip python3-vcstool python3-colcon-common-extensions \
    ros-${ROS_DISTRO}-rtabmap ros-${ROS_DISTRO}-rtabmap-ros \
    ros-${ROS_DISTRO}-pcl-ros ros-${ROS_DISTRO}-pcl-conversions \
    ros-${ROS_DISTRO}-octomap ros-${ROS_DISTRO}-octomap-msgs \
    ros-${ROS_DISTRO}-nav2-msgs ros-${ROS_DISTRO}-nav2-common \
    ros-${ROS_DISTRO}-tf2-eigen ros-${ROS_DISTRO}-tf2-geometry-msgs \
    ros-${ROS_DISTRO}-image-transport ros-${ROS_DISTRO}-image-transport-plugins \
    ros-${ROS_DISTRO}-laser-geometry ros-${ROS_DISTRO}-perception-pcl \
    ros-${ROS_DISTRO}-diagnostic-msgs ros-${ROS_DISTRO}-rviz2 \
    ros-${ROS_DISTRO}-rosbag2-storage-mcap \
    ros-${ROS_DISTRO}-foxglove-bridge \
    git wget curl unzip cmake \
    libboost-all-dev libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

# Build GTSAM from source
RUN git clone --depth 1 --branch 4.1.1 https://github.com/borglab/gtsam.git /tmp/gtsam && \
    cd /tmp/gtsam && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
             -DGTSAM_BUILD_TESTS=OFF \
             -DGTSAM_BUILD_UNSTABLE=ON \
             -DGTSAM_WITH_TBB=ON && \
    make -j$(nproc) && make install && ldconfig && \
    rm -rf /tmp/gtsam

# Python dependencies
# PyTorch with CUDA 12.x support
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir \
    "numpy<2" scipy scikit-learn \
    networkx numba open3d Pillow \
    sortedcontainers distinctipy numpy-quaternion evo

# TEASER++ for LiDAR geometric verification (optional but useful)
RUN pip3 install --no-cache-dir teaserpp_python || true

# Create non-root user with matching host UID/GID to avoid mount permission issues
RUN groupadd --gid ${USER_GID} ros \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ros \
    && echo "ros ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ros \
    && chmod 0440 /etc/sudoers.d/ros

# Auto-source ROS in every shell
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/ros/.bashrc && \
    echo '[ -f /home/ros/ws/install/setup.bash ] && source /home/ros/ws/install/setup.bash' >> /home/ros/.bashrc

# Auto-source ROS in every shell
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/ros/.bashrc && \
    echo '[ -f /home/ros/ws/install/setup.bash ] && source /home/ros/ws/install/setup.bash' >> /home/ros/.bashrc

WORKDIR /home/ros/ws
