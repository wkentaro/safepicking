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

from mercury.examples.picking._env import PickFromPileEnv
from mercury.examples.picking import _utils


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
    parser.add_argument("--nogui", action="store_true", help="no gui")
    parser.add_argument("--mp4", help="mp4")
    parser.add_argument("--noise", type=float, default=0.0, help="pose noise")
    parser.add_argument("--miss", type=float, default=0.0, help="pose miss")
    args = parser.parse_args()

    log_dir = here / f"logs/{args.planner}"

    if args.nogui:
        scene_id = args.pile_file.stem
        basename = f"eval-noise_{args.noise}-miss_{args.miss}"
        json_file = log_dir / f"{basename}/{scene_id}.json"
        if json_file.exists():
            logger.info(f"Result file already exists: {json_file}")
            return

    env = PickFromPileEnv(
        gui=not args.nogui,
        mp4=args.mp4,
        speed=0.005,
        pose_noise=args.noise,
        miss=args.miss,
        raise_on_timeout=True,
    )
    env.eval = True
    try:
        env.reset(
            random_state=np.random.RandomState(0),
            pile_file=args.pile_file,
        )
    except RuntimeError:
        with open(json_file, "w") as f:
            pass
        return

    ri = env.ri
    plane = env.plane
    object_ids = env.object_ids
    target_object_id = env.target_object_id

    ri.planner = args.planner

    with pp.WorldSaver():
        # prepare for planning
        is_missing_target_object = False
        for object_id, object_pose in zip(object_ids, env.object_state[2]):
            if (object_pose == 0).all():
                if object_id == target_object_id:
                    is_missing_target_object = True
                pp.set_pose(object_id, ([1, 1, 1], [0, 0, 0, 1]))
            else:
                pp.set_pose(object_id, (object_pose[:3], object_pose[3:]))
        if not is_missing_target_object:
            ee_to_world = ri.get_pose("tipLink")
            obj_to_world = pp.get_pose(target_object_id)
            obj_to_ee = pp.multiply(pp.invert(ee_to_world), obj_to_world)
            ri.attachments = [
                pp.Attachment(ri.robot, ri.ee, obj_to_ee, target_object_id)
            ]

        if ri.planner == "Heuristic":
            c = mercury.geometry.Coordinate(*ri.get_pose("tipLink"))
            steps = []
            for _ in range(5):
                c.translate([0, 0, 0.05], wrt="world")
                j = ri.solve_ik(c.pose)
                if j is not None:
                    steps.append(
                        ri.movej(j, speed=0.005, raise_on_timeout=True)
                    )
            steps.append(
                ri.movej(ri.homej, speed=0.005, raise_on_timeout=True)
            )
            steps = itertools.chain(*steps)
        else:
            assert len(ri.attachments) <= 1

            obstacles = [plane] + object_ids
            if ri.attachments:
                obstacles.remove(ri.attachments[0].child)

            for min_distance in np.linspace(0, -0.05, num=6):
                if ri.attachments:
                    min_distances = {
                        (ri.attachments[0].child, -1): min_distance
                    }
                else:
                    min_distances = None
                js = ri.planj(
                    ri.homej,
                    obstacles=obstacles,
                    min_distances=min_distances,
                )
                if js is not None:
                    break
                logger.warning(f"js is None w/ min_distance={min_distance}")
            else:
                js = [ri.homej]

            steps = []
            for j in js:
                steps.append(ri.movej(j, speed=0.005, raise_on_timeout=True))
            steps = itertools.chain(*steps)

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
            seed=0,
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
