#!/bin/bash

HERE=$(realpath $(dirname ${BASH_SOURCE[0]}))

setup () {
  # slurm script cannot use source command
  source ../../.anaconda3/bin/activate
}

cd $HERE
setup

./train.py $@
