#!/usr/bin/env python

import argparse

import numpy as np
import pybullet_planning as pp

from env import Env
from planned import plan_and_execute_place


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--class-ids", type=int, nargs="+", help="class ids", required=True
    )
    parser.add_argument("--mp4", help="mp4")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--timeout", type=float, default=3, help="timeout")
    args = parser.parse_args()

    env = Env(class_ids=args.class_ids, mp4=args.mp4)
    env.random_state = np.random.RandomState(args.seed)
    env.eval = True
    env.reset()

    with pp.LockRenderer():
        for object_id in env.object_ids:
            if object_id != env.fg_object_id:
                pp.remove_body(object_id)

        for _ in range(2400):
            pp.step_simulation()
    env.object_ids = [env.fg_object_id]
    env.update_obs()

    plan_and_execute_place(env)


if __name__ == "__main__":
    main()
