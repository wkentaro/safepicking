#!/usr/bin/env python

import argparse
import json
import pprint

from loguru import logger
import numpy as np
import path

import mercury

from _agent import DqnAgent
from _env import PickFromPileEnv
import _utils


here = path.Path(__file__).abspath().parent


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pile_file", type=path.Path, help="pile file")
    parser.add_argument(
        "--weight-dir", type=path.Path, help="weight dir", required=True
    )
    parser.add_argument("--nogui", action="store_true", help="no gui")
    parser.add_argument("--mp4", help="mp4")
    parser.add_argument("--noise", type=float, default=0.0, help="pose noise")
    parser.add_argument("--miss", type=float, default=0.0, help="pose miss")
    args = parser.parse_args()

    log_dir = args.weight_dir.parent.parent

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
        obs = env.reset(
            random_state=np.random.RandomState(0),
            pile_file=args.pile_file,
        )
    except RuntimeError:
        with open(json_file, "w") as f:
            pass
        return

    hparams_file = log_dir / "hparams.json"
    with open(hparams_file) as f:
        hparams = json.load(f)
    pprint.pprint(hparams)

    agent = DqnAgent(env=env, model=hparams["model"])
    agent.build(training=False)
    agent.load_weights(args.weight_dir)

    while True:
        for key in obs:
            obs[key] = obs[key][None, None, :]
        act_result = agent.act(
            step=-1, observation=obs, deterministic=True, env=env
        )
        transition = env.step(act_result)

        logger.info(
            f"reward={transition.reward}, terminal={transition.terminal}"
        )
        if transition.terminal:
            assert transition.reward == 1
            break
        else:
            obs = transition.observation

    translations = env.translations
    max_velocities = env.max_velocities

    for object_id in env.object_ids:
        if object_id == env.target_object_id:
            continue
        class_id = _utils.get_class_id(object_id)
        class_name = mercury.datasets.ycb.class_names[class_id]
        logger.info(
            f"[{object_id:2d}] {class_name:20s}: "
            f"translation={translations[object_id]:.2f}, "
            f"max_velocity={max_velocities[object_id]:.2f}, "
        )
    logger.info(f"sum_of_translations: {sum(translations.values()):.2f}")
    logger.info(f"sum_of_max_velocities: {sum(max_velocities.values()):.2f}")

    if args.nogui:
        data = dict(
            planner="Learned",
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
