#!/bin/bash

set -e

ROS_PREFIX=$HOME/ros_mercury
CONDA_PREFIX=$ROS_PREFIX/src/mercury/.anaconda3

source $CONDA_PREFIX/bin/activate
source /opt/ros/kinetic/setup.bash

set -x

# pip install catkin_pkg
# pip install defusedxml
# pip install rospkg
# pip install empy
# pip install netifaces
# pip install opencv-python

mkdir -p $ROS_PREFIX/src
cd $ROS_PREFIX
catkin init

catkin config --merge-devel \
              -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python \
              -DPYTHON_INCLUDE_DIR=$CONDA_PREFIX/include/python3.7m \
              -DPYTHON_LIBRARY=$CONDA_PREFIX/lib/libpython3.7m.so \
              --cmake-args -DCMAKE_BUILD_TYPE=Release -DOCTOMAP_OMP=1
catkin config --blacklist \
  roscpp \
  checkerboard_detector \
  jsk_network_tools \
  jsk_tools \
  jsk_recognition_msgs \
  imagesift \
  image_view2 \
  jsk_perception \
  jsk_pcl_ros \
  jsk_pcl_ros_utils

mkdir -p $ROS_PREFIX/devel/lib/python3/dist-packages
ln -fs $CONDA_PREFIX/lib/python3.7/site-packages/cv2 $ROS_PREFIX/devel/lib/python3/dist-packages

catkin build cv_bridge
catkin build tf
catkin build message_filters

unset PYTHONPATH

set +x
source $CONDA_PREFIX/bin/activate
source /opt/ros/kinetic/setup.bash
source $ROS_PREFIX/devel/setup.bash
set -x

python -c 'import cv2'
python -c 'from cv_bridge.boost.cv_bridge_boost import getCvType'
python -c 'import tf'

catkin build realsense2_camera
catkin build morefusion_panda_ycb_video
