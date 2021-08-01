#!/bin/bash -e

HERE=$(realpath $(dirname ${BASH_SOURCE[0]}))
source $HERE/__init__.sh
ROOT=$(realpath $HERE/..)

source $ROOT/.anaconda3/bin/activate

if [ -e $ROOT/src/ompl/py-bindings/ompl/base/_base.so ]; then
  echo_bold "==> OMPL is already installed"
  exit 0
fi

echo_bold "==> Installing dependencies"

conda_install cmake boost==1.71.0 eigen

pip_install pyplusplus castxml

echo_bold "==> Installing OMPL"

mkdir -p $ROOT/src
cd $ROOT/src
if [ ! -d ompl ]; then
  git clone https://github.com/ompl/ompl.git
fi
cd ompl
git checkout 1.5.1

mkdir -p build
cd build
cmake .. -DOMPL_BUILD_PYBINDINGS=TRUE -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
make -j 4 update_bindings
make -j 4
