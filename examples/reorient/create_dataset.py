#!/usr/bin/env python

from loguru import logger
import numpy as np
import path

from pick_and_place_env import PickAndPlaceEnv
from planned import rollout_plan_reorient


home = path.Path("~").expanduser()


def main():
    env = PickAndPlaceEnv(gui=False)
    env.random_state = np.random.RandomState(5)
    env.reset(pile_file=env.PILES_DIR / "00001000.npz")

    for i, result in enumerate(rollout_plan_reorient(env)):
        npz_file = home / f"data/mercury/reorient/00001000/seed5/{i:08d}.npz"
        npz_file.parent.makedirs_p()
        np.savez_compressed(
            npz_file,
            **dict(
                grasp_pose=np.hstack(result["c_grasp"].pose),
                reorient_pose=np.hstack(result["c_reorient"].pose),
                js_place_length=result["js_place_length"],
            ),
        )
        logger.info(f"Saved to: {npz_file}")


if __name__ == "__main__":
    main()
