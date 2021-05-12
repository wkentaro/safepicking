#!/usr/bin/env python

import time

import numpy as np
import path
import pybullet_planning as pp

import mercury

from pick_and_place_env import PickAndPlaceEnv


home = path.Path("~").expanduser()


def main():
    env = PickAndPlaceEnv(gui=True)
    env.random_state = np.random.RandomState(5)
    env.reset(pile_file=env.PILES_DIR / "00001000.npz")

    root_dir = home / "data/mercury/reorient/00001000/seed5"
    for npz_file in sorted(root_dir.listdir()):
        result = np.load(npz_file)

        grasp_pose = result["grasp_pose"]
        reorient_pose = result["reorient_pose"]
        js_place_length = result["js_place_length"]

        print(npz_file, js_place_length)

        j = env.ri.solve_ik((grasp_pose[:3], grasp_pose[3:]))
        env.ri.setj(j)

        obj_af = mercury.pybullet.duplicate(
            env.fg_object_id,
            collision=False,
            texture=False,
            rgba_color=(0, 1, 0, 0.5),
            position=reorient_pose[:3],
            quaternion=reorient_pose[3:],
        )

        time.sleep(1)

        pp.remove_body(obj_af)


if __name__ == "__main__":
    main()
