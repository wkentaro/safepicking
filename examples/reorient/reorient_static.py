#!/usr/bin/env python

import argparse
import itertools
import json
import time

from loguru import logger
import numpy as np
import path
import pybullet_planning as pp

from _env import Env
import _reorient


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, help="seed", required=True)
    parser.add_argument(
        "--face",
        choices=["front", "right", "left"],
        default="front",
        help="face",
    )
    parser.add_argument("--mp4", help="mp4")
    parser.add_argument("--nogui", action="store_true", help="no gui")
    args = parser.parse_args()

    json_file = path.Path(
        f"logs/reorient_static/{args.seed:08d}-{args.face}.json"
    )
    if args.nogui and json_file.exists():
        logger.info(f"Already json_file exists: {json_file}")
        return

    env = Env(
        class_ids=[2, 3, 5, 11, 12, 15],
        mp4=args.mp4,
        gui=not args.nogui,
        face=args.face,
    )
    env.random_state = np.random.RandomState(args.seed)
    env.eval = True
    env.reset()

    reorient_poses = _reorient.get_static_reorient_poses(env)
    grasp_poses = np.array(
        list(itertools.islice(_reorient.get_grasp_poses(env), 100))
    )

    for reorient_pose in reorient_poses:
        p = np.random.permutation(grasp_poses.shape[0])[:3]
        for grasp_pose in grasp_poses[p]:
            result = _reorient.plan_reorient(env, grasp_pose, reorient_pose)
            if "js_place" in result:
                break
        if "js_place" in result:
            break

    if "js_place" not in result:
        logger.error("No solution is found")
        success = False
        execution_time = np.nan
        trajectory_length = np.nan
    else:
        exec_result = _reorient.execute_reorient(env, result)
        execution_time = exec_result["t_place"]
        trajectory_length = result["js_place_length"]

        for _ in range(480):
            pp.step_simulation()
            if pp.has_gui():
                time.sleep(pp.get_time_step())

        success = _reorient.plan_and_execute_place(env)

    if args.nogui:
        json_file.parent.makedirs_p()
        with open(json_file, "w") as f:
            json.dump(
                dict(
                    success=success,
                    trajectory_length=trajectory_length,
                    execution_time=execution_time,
                ),
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
