#!/usr/bin/env python

import argparse
import json
import time

from loguru import logger
import numpy as np
import path
import pybullet as p
import pybullet_planning as pp

import common_utils
from env import PickFromPileEnv


here = path.Path(__file__).abspath().parent


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("export_file", type=path.Path, help="export file")
    parser.add_argument(
        "--planner",
        choices=["RRTConnect", "Naive"],
        required=True,
        help="planner",
    )
    parser.add_argument("--pose-noise", action="store_true", help="pose noise")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--nogui", action="store_true", help="no gui")
    args = parser.parse_args()

    log_dir = here / f"logs/{args.planner}"
    scene_id = args.export_file.stem

    if args.nogui:
        json_file = (
            log_dir
            / f"eval-pose_noise_{args.pose_noise}/{scene_id}/{args.seed}.json"
        )
        if json_file.exists():
            logger.info(f"Result file already exists: {json_file}")
            return

    env = PickFromPileEnv(
        gui=not args.nogui,
        retime=10,
        planner=args.planner,
        pose_noise=args.pose_noise,
        suction_max_force=None,
        reward="max_velocities",
    )
    env.eval = True
    obs = env.reset(
        random_state=np.random.RandomState(args.seed),
        pile_file=args.export_file,
    )

    ri = env.ri
    plane = env.plane
    object_ids = env.object_ids
    target_object_id = env.target_object_id

    with pp.LockRenderer(), pp.WorldSaver():
        for i in range(len(object_ids)):
            pose = (
                obs["object_poses_openloop"][i, :3],
                obs["object_poses_openloop"][i, 3:],
            )
            pp.set_pose(object_ids[i], pose)
        steps = ri.move_to_homej(
            bg_object_ids=[plane],
            object_ids=object_ids,
            speed=0.001,
            timeout=30,
        )

    max_velocities = {}
    for _ in steps:
        p.stepSimulation()
        if args.suction_max_force is not None:
            ri.step_simulation()
        if not args.nogui:
            time.sleep(pp.get_time_step() / env._retime)

        for object_id in object_ids:
            if object_id == target_object_id:
                continue
            max_velocities[object_id] = max(
                max_velocities.get(object_id, 0),
                np.linalg.norm(pp.get_velocity(object_id)[0]),
            )

    success = ri.gripper.check_grasp()
    if success:
        logger.success("Task is complete")
    else:
        logger.error("Task is failed")

    for object_id in object_ids:
        if object_id == target_object_id:
            continue
        logger.info(
            f"object_id={object_id}, "
            f"class_id={common_utils.get_class_id(object_id):02d}, "
            f"max_velocity={max_velocities[object_id]:.3f}"
        )
    logger.info(f"sum_of_max_velocities: {sum(max_velocities.values()):.3f}")

    if args.nogui:
        data = dict(
            planner=args.planner,
            scene_id=scene_id,
            seed=args.seed,
            success=success,
            max_velocities=list(max_velocities.values()),
            sum_of_max_velocities=sum(max_velocities.values()),
        )

        json_file.parent.makedirs_p()
        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved to: {json_file}")


if __name__ == "__main__":
    main()
