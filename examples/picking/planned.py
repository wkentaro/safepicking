#!/usr/bin/env python

import argparse
import collections
import itertools
import json
import time

from loguru import logger
import numpy as np
import path
import pybullet as p
import pybullet_planning as pp

import mercury

from _env import PickFromPileEnv
import _utils


here = path.Path(__file__).abspath().parent


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pile_file", type=path.Path, help="pile file")
    parser.add_argument(
        "--planner",
        choices=["RRTConnect", "Heuristic", "Naive"],
        required=True,
        help="planner",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--nogui", action="store_true", help="no gui")
    parser.add_argument("--mp4", help="mp4")
    parser.add_argument(
        "--pose-noise", type=float, default=0.0, help="pose noise"
    )
    parser.add_argument("--miss", type=float, default=0.2, help="miss")
    args = parser.parse_args()

    log_dir = here / f"logs/{args.planner}"

    if args.nogui:
        scene_id = args.pile_file.stem
        json_file = (
            log_dir
            / f"eval-noise_{args.pose_noise}-miss_{args.miss}/{scene_id}/{args.seed}.json"  # NOQA
        )
        if json_file.exists():
            logger.info(f"Result file already exists: {json_file}")
            return

    env = PickFromPileEnv(
        gui=not args.nogui,
        mp4=args.mp4,
        pose_noise=args.pose_noise,
        miss=args.miss,
    )
    env.eval = True
    env.reset(
        random_state=np.random.RandomState(args.seed),
        pile_file=args.pile_file,
    )

    ri = env.ri
    plane = env.plane
    object_ids = env.object_ids
    target_object_id = env.target_object_id

    ri.planner = args.planner

    with pp.LockRenderer(), pp.WorldSaver():
        for object_id, object_pose in zip(object_ids, env.object_state[2]):
            if (object_pose == 0).all():
                pp.set_pose(object_id, ([1, 1, 1], [0, 0, 0, 1]))
            else:
                pp.set_pose(object_id, (object_pose[:3], object_pose[3:]))
        if ri.planner == "Heuristic":
            c = mercury.geometry.Coordinate(*ri.get_pose("tipLink"))
            steps = []
            for _ in range(5):
                c.translate([0, 0, 0.05], wrt="world")
                j = ri.solve_ik(c.pose)
                if j is not None:
                    steps.append(ri.movej(j, speed=0.005))
            steps.append(ri.movej(ri.homej, speed=0.005))
            steps = itertools.chain(*steps)
        else:
            steps = ri.move_to_homej(
                bg_object_ids=[plane],
                object_ids=object_ids,
                speed=0.005,
                timeout=20,
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
        step_callback()
        if not args.nogui:
            time.sleep(pp.get_time_step())

    for object_id in object_ids:
        if object_id == target_object_id:
            continue
        class_id = _utils.get_class_id(object_id)
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
            target_object_class=int(env.target_object_class),
            target_object_visibility=float(env.target_object_visibility),
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