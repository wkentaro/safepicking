#!/bin/bash -e

HERE=$(realpath $(dirname ${BASH_SOURCE[0]}))
source $HERE/__init__.sh
ROOT=$(realpath $HERE/..)

source $ROOT/.anaconda3/bin/activate

if [ -e $ROOT/src/ompl/py-bindings/ompl/base/_base.so ]; then
  echo_bold "==> OMPL is already installed"
  exit 0
fi

echo_bold "==> Installing OMPL"

cd $ROOT/src
git clone https://github.com/ompl/ompl.git
cd ompl

pip install pyplusplus

mkdir -p build
cmake .. -DOMPL_BUILD_PYBINDINGS=TRUE
make -j update_bindings
make -j