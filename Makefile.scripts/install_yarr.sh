#!/bin/bash -e

HERE=$(realpath $(dirname ${BASH_SOURCE[0]}))
source $HERE/__init__.sh
ROOT=$(realpath $HERE/..)

source $ROOT/.anaconda3/bin/activate

echo_bold "==> Installing YARR"

mkdir -p $ROOT/src
cd $ROOT/src
if [ ! -d _YARR ]; then
  git clone https://github.com/stepjam/_YARR.git
fi
cd _YARR

git checkout 82d84d6

pip_install -r requirements.txt
pip_install natsort
pip_install -e .
