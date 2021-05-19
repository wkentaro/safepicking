#!/usr/bin/env python

import argparse
import json
import pprint

from loguru import logger
import numpy as np
import path

from agent import DqnAgent
from env import PickFromPileEnv

import common_utils


here = path.Path(__file__).abspath().parent


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("export_file", type=path.Path, help="export file")
    parser.add_argument(
        "--weight-dir", type=path.Path, help="weight dir", required=True
    )
    parser.add_argument("--pose-noise", action="store_true", help="pose noise")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--nogui", action="store_true", help="no gui")
    parser.add_argument("--force", action="store_true", help="force")
    args = parser.parse_args()

    log_dir = args.weight_dir.parent.parent
    scene_id = args.export_file.stem
    json_file = (
        log_dir
        / f"eval-pose_noise_{args.pose_noise}/{scene_id}/{args.seed}.json"
    )

    if not args.force and json_file.exists():
        logger.info(f"Result file already exists: {json_file}")
        return

    hparams_file = log_dir / "hparams.json"
    with open(hparams_file) as f:
        hparams = json.load(f)

    pprint.pprint(hparams)

    env = PickFromPileEnv(
        gui=not args.nogui,
        retime=10,
        planner="RRTConnect",
        pose_noise=args.pose_noise,
        suction_max_force=hparams["suction_max_force"],
        reward=hparams["reward"],
    )
    env.eval = True
    obs = env.reset(
        random_state=np.random.RandomState(args.seed),
        pile_file=args.export_file,
    )

    agent = DqnAgent(env=env, model=hparams["model"])
    agent.build(training=False)
    agent.load_weights(args.weight_dir)

    ri = env.ri
    object_ids = env.object_ids
    target_object_id = env.target_object_id

    max_velocities = {}

    while True:
        for key in obs:
            obs[key] = obs[key][None, None, :]
        act_result = agent.act(
            step=-1, observation=obs, deterministic=True, env=env
        )
        transition = env.step(act_result)

        for object_id, max_velocity in transition.info[
            "max_velocities"
        ].items():
            max_velocities[object_id] = max(
                max_velocities.get(object_id, 0), max_velocity
            )

        print(
            f"action={act_result.action}, reward={transition.reward}, "
            f"terminal={transition.terminal}",
        )
        if transition.terminal:
            break
        else:
            obs = transition.observation

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

    data = dict(
        planner="Learned",
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
