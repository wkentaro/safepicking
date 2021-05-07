#!/usr/bin/env python

import argparse
import json
import time

import imgviz
import path

import mercury

from agent import DqnAgent
from env import GraspWithIntentEnv


here = path.Path(__file__).abspath().parent


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pile_file", type=path.Path, help="pile file")
    parser.add_argument("--weight-dir", type=path.Path, help="weight dir")
    parser.add_argument("--nogui", action="store_true", help="no gui")
    args = parser.parse_args()

    if args.weight_dir is None:
        log_dir = here / "logs/Naive"
        model = "rgbd"
    else:
        log_dir = args.weight_dir.parent.parent
        with open(log_dir / "hparams.json") as f:
            hparams = json.load(f)
        model = hparams["model"]

    retime = 10

    class StepCallback:
        def __init__(self):
            self._i = 0

        def __call__(self):
            if self._i % (8 * retime) == 0:
                rgb, _, _ = mercury.pybullet.get_debug_visualizer_image()
                rgb_file = (
                    log_dir
                    / f"eval/{args.pile_file.stem}/video/{self._i:08d}.jpg"
                )
                rgb_file.parent.makedirs_p()
                imgviz.io.imsave(rgb_file, rgb)
            self._i += 1

    env = GraspWithIntentEnv(
        gui=not args.nogui, retime=retime, step_callback=StepCallback()
    )
    env.eval = True
    obs = env.reset(pile_file=args.pile_file)

    agent = DqnAgent(
        validate_exploration=True, num_validate=None, env=env, model=model
    )
    agent.build(training=False)

    if args.weight_dir is not None:
        agent.load_weights(args.weight_dir)

    planning_times = []

    while True:
        for key in obs:
            obs[key] = obs[key][None, None, :]
        t_start = time.time()
        act_result = agent.act(
            step=-1,
            observation=obs,
            deterministic=args.weight_dir is not None,
            env=env,
        )
        planning_time = time.time() - t_start

        if 1:
            imgviz.io.cv_imshow(agent.draw_act_summary(), "act_summary")
            imgviz.io.cv_waitkey(100)

        transition = env.step(act_result)
        print(
            f"action={act_result.action}, reward={transition.reward}, "
            f"terminal={transition.terminal}",
        )
        if transition.reward == 1:
            planning_times.append(planning_time)
        if transition.info["needs_reset"]:
            break
        else:
            obs = transition.observation

        json_file = log_dir / f"eval/{args.pile_file.stem}/planning_times.json"
        json_file.parent.makedirs_p()
        with open(json_file, "w") as f:
            json.dump(dict(planning_times=planning_times), f, indent=2)


if __name__ == "__main__":
    main()
