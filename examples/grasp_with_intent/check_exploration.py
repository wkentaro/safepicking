#!/usr/bin/env python

from loguru import logger
import path

from agent import DqnAgent
from env import PickAndPlaceEnv


home = path.Path("~").expanduser()


def main():
    env = PickAndPlaceEnv()
    obs = env.reset()

    agent = DqnAgent(env=env, model="rgb", imshow=False)
    agent.build(training=False)

    while True:
        for key in obs:
            obs[key] = obs[key][None, None, :]
        act_result = agent.act(
            step=-1, observation=obs, deterministic=False, env=env
        )

        transition = env.step(act_result=act_result)

        logger.info(f"reward={transition.reward}")

        if transition.info["needs_reset"]:
            obs = env.reset()
        else:
            obs = transition.observation


if __name__ == "__main__":
    main()
