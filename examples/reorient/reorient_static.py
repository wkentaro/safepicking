#!/usr/bin/env python

import argparse
import itertools

import numpy as np

import mercury

from _env import Env
import _reorient


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, help="seed", required=True)
    parser.add_argument("--mp4", help="mp4")
    args = parser.parse_args()

    env = Env(class_ids=[2, 3, 5, 11, 12, 15], mp4=args.mp4)
    env.random_state = np.random.RandomState(args.seed)
    env.eval = True
    env.reset()

    # if _reorient.plan_and_execute_place(env):
    #     return

    reorient_poses = _reorient.get_static_reorient_poses(env)
    grasp_poses = np.array(
        list(itertools.islice(_reorient.get_grasp_poses(env), 100))
    )

    for reorient_pose in reorient_poses:
        c_reorient = mercury.geometry.Coordinate(
            *np.hsplit(reorient_pose, [3])
        )

        p = np.random.permutation(grasp_poses.shape[0])[:3]
        for grasp_pose in grasp_poses[p]:
            c_grasp = mercury.geometry.Coordinate(*np.hsplit(grasp_pose, [3]))

            result = _reorient.plan_reorient(env, c_grasp, c_reorient)

            if "js_place" in result:
                _reorient.execute_plan(env, result)
                return


if __name__ == "__main__":
    main()
