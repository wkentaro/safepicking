#!/usr/bin/env python

import argparse
import collections
import json
import time

from loguru import logger
import numpy as np
import path
import pybullet as p
import pybullet_planning as pp

import mercury

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

    poses = {}
    for object_id in object_ids:
        if object_id == target_object_id:
            continue
        poses[object_id] = pp.get_pose(object_id)
    translations = collections.defaultdict(int)
    max_velocities = collections.defaultdict(int)

    def step_callback():
        for object_id in object_ids:
            if object_id == target_object_id:
                continue

            pose = pp.get_pose(object_id)
            translations[object_id] += np.linalg.norm(
                np.array(poses[object_id][0]) - np.array(pose[0])
            )
            poses[object_id] = pose

            max_velocities[object_id] = max(
                max_velocities[object_id],
                np.linalg.norm(pp.get_velocity(object_id)[0]),
            )

    for _ in steps:
        p.stepSimulation()
        if env._suction_max_force is not None:
            ri.step_simulation()
        step_callback()
        if not args.nogui:
            time.sleep(pp.get_time_step() / env._retime)

    success = ri.gripper.check_grasp()
    if success:
        logger.success("Task is complete")
    else:
        logger.error("Task is failed")

    for object_id in object_ids:
        if object_id == target_object_id:
            continue
        class_id = common_utils.get_class_id(object_id)
        class_name = mercury.datasets.ycb.class_names[class_id]
        logger.info(
            f"[{object_id}] {class_name:20s}: "
            f"translation={translations[object_id]:.2f}, "
            f"max_velocity={max_velocities[object_id]:.2f}"
        )
    logger.info(f"sum_of_translations: {sum(translations.values()):.2f}")
    logger.info(f"sum_of_max_velocities: {sum(max_velocities.values()):.2f}")

    if args.nogui:
        data = dict(
            planner=args.planner,
            scene_id=scene_id,
            seed=args.seed,
            success=success,
            translations=dict(translations),
            sum_of_translations=sum(translations.values()),
            max_velocities=dict(max_velocities),
            sum_of_max_velocities=sum(max_velocities.values()),
        )

        json_file.parent.makedirs_p()
        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved to: {json_file}")


if __name__ == "__main__":
    main()
