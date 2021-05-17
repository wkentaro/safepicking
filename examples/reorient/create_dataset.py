#!/usr/bin/env python

import argparse

from loguru import logger
import numpy as np
import path
import pybullet_planning as pp

import mercury

from pick_and_place_env import PickAndPlaceEnv
from planned import rollout_plan_reorient


home = path.Path("~").expanduser()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gui", action="store_true", help="gui")
    args = parser.parse_args()

    seed = 5

    env = PickAndPlaceEnv(gui=args.gui)
    env.random_state = np.random.RandomState(seed)
    env.reset(pile_file=env.PILES_DIR / "00001000.npz")

    for i, result in enumerate(
        rollout_plan_reorient(
            env,
            return_failed=True,
            reorient_num_delta=32,
            reorient_num_sample=4,
            reorient_centroid=False,
            reorient_discretize=False,
            grasp_num_sample=4,
        )
    ):
        if i > 9999:
            break

        if "js_place_length" in result:
            with pp.WorldSaver():
                pp.set_pose(env.fg_object_id, result["c_reorient"].pose)
                for _ in range(int(1 / pp.get_time_step())):
                    pp.step_simulation()
                pose_actual = pp.get_pose(env.fg_object_id)

            pcd = np.loadtxt(
                mercury.datasets.ycb.get_pcd_file(class_id=env.FG_CLASS_ID)
            )
            reference = mercury.geometry.transform_points(
                pcd,
                mercury.geometry.transformation_matrix(
                    *result["c_reorient"].pose
                ),
            )
            query = mercury.geometry.transform_points(
                pcd, mercury.geometry.transformation_matrix(*pose_actual)
            )
            auc = mercury.geometry.average_distance_auc(
                reference, query, max_threshold=0.2
            )
        else:
            auc = np.nan

        npz_file = (
            home / f"data/mercury/reorient/00001000/{seed:04d}_{i:04d}.npz"
        )
        npz_file.parent.makedirs_p()
        np.savez_compressed(
            npz_file,
            **dict(
                grasp_pose=np.hstack(result["c_grasp"].pose),
                reorient_pose=np.hstack(result["c_reorient"].pose),
                js_place_length=result.get("js_place_length", np.nan),
                auc=auc,
            ),
        )
        logger.info(f"Saved to: {npz_file}")


if __name__ == "__main__":
    main()
