#!/bin/bash

waitforjobs() {
    while test $(jobs -p | wc -w) -ge "$1"; do wait -n; done
}

for scene_id in $(seq 9000 9999); do
  scene_id=$(printf "%08d" $scene_id)
  pile_file=data/pile_generation/${scene_id}.pkl

  set -x

  weight_dir_conv_net=logs/20210706_194543-conv_net/weights/91500
  weight_dir_openloop_pose_net=logs/20210709_005731-openloop_pose_net-noise/weights/90500
  weight_dir_fusion_net=logs/20210709_005731-fusion_net-noise/weights/84500

  ./planned.py $pile_file --nogui --planner Heuristic &
  ./planned.py $pile_file --nogui --planner Naive &
  ./learned.py $pile_file --nogui --weight-dir $weight_dir_conv_net &

  # for miss in 0.5 0.4 0.3 0.2 0.0; do
  #   ./planned.py --noise 0.3 --miss $miss $pile_file --nogui --planner RRTConnect &
  #   ./learned.py --noise 0.3 --miss $miss $pile_file --nogui --weight-dir $weight_dir_openloop_pose_net &
  #   ./learned.py --noise 0.3 --miss $miss $pile_file --nogui --weight-dir $weight_dir_fusion_net &
  # done

  { set +x; } 2>/dev/null

  waitforjobs 12
done
wait
