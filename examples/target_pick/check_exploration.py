#!/usr/bin/env python

import argparse
import pprint
import time

from loguru import logger
import numpy as np
from yarr.agents.agent import ActResult

from agent import DqnAgent
from env import PickFromPileEnv


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["closedloop_pose_net"],
        help="model",
    )
    parser.add_argument("--nogui", action="store_true", help="no gui")
    parser.add_argument("--print-obs", action="store_true", help="print obs")
    args = parser.parse_args()

    env = PickFromPileEnv(gui=not args.nogui)
    t_start = time.time()
    obs = env.reset()
    logger.info(f"Reset time: {time.time() - t_start:.2f} [s]")
    i = 0
    if args.print_obs:
        pprint.pprint(obs)

    if args.model is None:
        agent = None
    else:
        agent = DqnAgent(env=env, model=args.model)
        agent.build(training=False)

    while True:
        for key in obs:
            obs[key] = obs[key][None, None, :]
        if agent is None:
            if 1:
                act_result = env.get_demo_action()
            else:
                for action in np.random.permutation(len(env.actions)):
                    act_result = ActResult(action)
                    if env.validate_action(act_result):
                        break
        else:
            act_result = agent.act(
                step=-1, observation=obs, deterministic=False, env=env
            )
        t_start = time.time()
        transition = env.step(act_result)
        logger.info(f"[{i}] Step time: {time.time() - t_start:.2f} [s]")
        if transition.terminal:
            t_start = time.time()
            obs = env.reset()
            logger.info(f"Reset time: {time.time() - t_start:.2f} [s]")
            i = 0
        else:
            obs = transition.observation
            i += 1


if __name__ == "__main__":
    main()
