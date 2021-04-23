#!/usr/bin/env python

import pprint

import numpy as np

from yarr.agents.agent import ActResult

from env import PickFromPileEnv


def agent(env, obs):
    for action in np.random.permutation(len(env.actions)):
        act_result = ActResult(action=action)
        if env.validate_action(act_result):
            break
    return act_result


def main():
    env = PickFromPileEnv(gui=True, retime=10)
    obs = env.reset()
    pprint.pprint(obs)

    while True:
        act_result = agent(env, obs)
        transition = env.step(act_result)
        print(
            f"action={act_result.action}, reward={transition.reward}, ",
            f"terminal={transition.terminal}",
        )
        if transition.terminal:
            obs = env.reset()
        else:
            obs = transition.observation


if __name__ == "__main__":
    main()
