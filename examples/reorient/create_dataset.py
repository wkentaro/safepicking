#!/usr/bin/env python

import argparse
import itertools

from loguru import logger
import numpy as np
import path
import pybullet_planning as pp

import mercury

import common_utils
from pick_and_place_env import PickAndPlaceEnv
from planned import rollout_plan_reorient


home = path.Path("~").expanduser()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gui", action="store_true", help="gui")
    parser.add_argument(
        "--process-id", type=str, required=True, help="process id (e.g., 2/5)"
    )
    parser.add_argument("--size", type=int, default=50000, help="dataset size")
    args = parser.parse_args()

    process_index, process_num = args.process_id.split("/")
    process_index = int(process_index)
    process_num = int(process_num)

    env = PickAndPlaceEnv(gui=args.gui)

    i = process_index
    while True:
        env.reset()

        for result in itertools.islice(
            rollout_plan_reorient(
                env,
                return_failed=True,
                reorient_discretize=False,
            ),
            10,
        ):
            if "js_place_length" in result:
                with pp.WorldSaver():
                    pp.set_pose(env.fg_object_id, result["c_reorient"].pose)
                    for _ in range(int(1 / pp.get_time_step())):
                        pp.step_simulation()
                    pose_actual = pp.get_pose(env.fg_object_id)

                pcd = np.loadtxt(
                    mercury.datasets.ycb.get_pcd_file(
                        class_id=common_utils.get_class_id(env.fg_object_id)
                    )
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

            object_fg_flags = []
            object_classes = []
            object_poses = []
            for object_id in env.object_ids:
                object_fg_flags.append(object_id == env.fg_object_id)
                object_classes.append(common_utils.get_class_id(object_id))
                object_poses.append(np.hstack(pp.get_pose(object_id)))

            while True:
                npz_file = (
                    home / f"data/mercury/reorient/n_class_5/{i:08d}.npz"
                )

                if not npz_file.exists():
                    npz_file.parent.makedirs_p()
                    np.savez_compressed(
                        npz_file,
                        **dict(
                            object_fg_flags=object_fg_flags,
                            object_classes=object_classes,
                            object_poses=object_poses,
                            grasp_pose=np.hstack(result["c_grasp"].pose),
                            initial_pose=np.hstack(result["c_init"].pose),
                            reorient_pose=np.hstack(result["c_reorient"].pose),
                            js_place_length=result.get(
                                "js_place_length", np.nan
                            ),
                            auc=auc,
                        ),
                    )
                    logger.info(f"Saved to: {npz_file}")
                    break

                i += process_num
                if i >= args.size:
                    return


if __name__ == "__main__":
    main()
