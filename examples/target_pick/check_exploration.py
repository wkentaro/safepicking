#!/usr/bin/env python

import pprint

from agent import DqnAgent
from env import PickFromPileEnv


def main():
    env = PickFromPileEnv(gui=True, retime=10)
    obs = env.reset()
    pprint.pprint(obs)

    agent = DqnAgent(env=env)
    agent.build(training=False)

    while True:
        for key in obs:
            obs[key] = obs[key][None, None, :]
        act_result = agent.act(
            step=-1, observation=obs, deterministic=False, env=env
        )
        transition = env.step(act_result)
        print(
            f"action={act_result.action}, reward={transition.reward}, "
            f"terminal={transition.terminal}",
        )
        if transition.terminal:
            obs = env.reset()
        else:
            obs = transition.observation


if __name__ == "__main__":
    main()
