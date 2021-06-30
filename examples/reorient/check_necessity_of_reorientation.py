#!/usr/bin/env python

import argparse
import sys

import numpy as np

from _env import Env
import _reorient


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, help="seed", required=True)
    parser.add_argument(
        "--face",
        choices=["front", "right", "left"],
        default="front",
        help="face",
    )
    parser.add_argument("--mp4", help="mp4")
    parser.add_argument("--nogui", action="store_true", help="no gui")
    args = parser.parse_args()

    env = Env(
        class_ids=[2, 3, 5, 11, 12, 15],
        gui=not args.nogui,
        mp4=args.mp4,
        face=args.face,
    )
    env.random_state = np.random.RandomState(args.seed)
    env.eval = True
    env.reset()

    pick_and_placable = _reorient.plan_and_execute_place(env)
    # 1: pick_and_placable=True, necessity_of_reorientation=False
    # 0: pick_and_placable=False, necessity_of_reorientation=True
    sys.exit(int(pick_and_placable))


if __name__ == "__main__":
    main()
