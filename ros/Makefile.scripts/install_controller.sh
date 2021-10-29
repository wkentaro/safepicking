#!/bin/bash

HERE=$(realpath $(dirname ${BASH_SOURCE:-$0}))
ROOT=$(realpath $HERE/../..)

CATKIN_WS=$ROOT/ros

set -e

source /opt/ros/kinetic/setup.bash

set -x

cd $CATKIN_WS

mkdir -p src/
catkin init
catkin build
