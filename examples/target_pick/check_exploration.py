#!/usr/bin/env python

import argparse
import pprint
import time

from loguru import logger

from agent import DqnAgent
from env import PickFromPileEnv


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["closedloop_pose_net", "openloop_pose_net"],
        help="model",
    )
    parser.add_argument("--nogui", action="store_true", help="no gui")
    parser.add_argument("--print-obs", action="store_true", help="print obs")
    args = parser.parse_args()

    env = PickFromPileEnv(gui=not args.nogui, retime=10)
    t_start = time.time()
    obs = env.reset()
    logger.info(f"Reset time: {time.time() - t_start:.2f} [s]")
    if args.print_obs:
        pprint.pprint(obs)

    agent = DqnAgent(env=env, model=args.model)
    agent.build(training=False)

    while True:
        for key in obs:
            obs[key] = obs[key][None, None, :]
        act_result = agent.act(
            step=-1, observation=obs, deterministic=False, env=env
        )
        t_start = time.time()
        transition = env.step(act_result)
        logger.info(f"Step time: {time.time() - t_start:.2f} [s]")
        logger.info(
            f"action={act_result.action}, reward={transition.reward}, "
            f"terminal={transition.terminal}",
        )
        if transition.terminal:
            t_start = time.time()
            obs = env.reset()
            logger.info(f"{time.time() - t_start:.2f} [s]")
        else:
            obs = transition.observation


if __name__ == "__main__":
    main()
