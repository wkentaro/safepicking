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
    parser.add_argument("--force", action="store_true", help="force")
    args = parser.parse_args()

    log_dir = here / f"logs/{args.planner}"
    scene_id = args.export_file.stem
    json_file = (
        log_dir
        / f"eval-pose_noise_{args.pose_noise}/{scene_id}/{args.seed}.json"
    )

    if not args.force and json_file.exists():
        logger.info(f"Result file already exists: {json_file}")
        return

    env = PickFromPileEnv(
        gui=not args.nogui, planner=args.planner, pose_noise=args.pose_noise
    )
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

    velocities = {}
    for _ in steps:
        p.stepSimulation()
        ri.step_simulation()
        if not args.nogui:
            time.sleep(1 / 240)

        for object_id in object_ids:
            if object_id == target_object_id:
                continue
            velocities[object_id] = max(
                velocities.get(object_id, 0),
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
            f"velocity={velocities[object_id]:.3f}"
        )
    logger.info(f"sum_of_velocities: {sum(velocities.values()):.3f}")

    data = dict(
        planner=args.planner,
        scene_id=scene_id,
        seed=args.seed,
        success=success,
        velocities=list(velocities.values()),
        sum_of_velocities=sum(velocities.values()),
    )

    json_file.parent.makedirs_p()
    with open(json_file, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"saved to: {json_file}")


if __name__ == "__main__":
    main()
