#!/usr/bin/env python

import argparse
import itertools
import pickle

from loguru import logger
import numpy as np
import path
import pybullet_planning as pp

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
    parser.add_argument(
        "--size", type=int, default=150000, help="dataset size"
    )
    parser.add_argument(
        "--class-ids", type=int, nargs="+", help="class ids", required=True
    )
    args = parser.parse_args()

    args.class_ids = sorted(args.class_ids)

    process_index, process_num = args.process_id.split("/")
    process_index = int(process_index)
    process_num = int(process_num)

    env = PickAndPlaceEnv(class_ids=args.class_ids, gui=args.gui)

    i = process_index
    while True:
        env.reset()

        for result in itertools.islice(
            rollout_plan_reorient(
                env,
                return_failed=True,
                grasp_num_sample=1,
                min_angle=np.deg2rad(0),
                max_angle=np.deg2rad(95),
            ),
            10,
        ):
            object_fg_flags = []
            object_classes = []
            object_poses = []
            for object_id in env.object_ids:
                object_fg_flags.append(object_id == env.fg_object_id)
                object_classes.append(common_utils.get_class_id(object_id))
                object_poses.append(np.hstack(pp.get_pose(object_id)))
            object_fg_flags = np.array(object_fg_flags, dtype=bool)
            object_classes = np.array(object_classes, dtype=np.int32)
            object_poses = np.array(object_poses, dtype=np.float32)

            keys = [
                "j_grasp",
                "j_place",
                "js_grasp",
                "js_pre_grasp",
                "js_place",
            ]
            solved = np.array([key in result for key in keys], dtype=bool)

            ee_to_world = result["c_grasp"].pose
            obj_to_world = result["c_init"].pose
            ee_to_obj = pp.multiply(pp.invert(obj_to_world), ee_to_world)

            obj_af_to_world = result["c_reorient"].pose

            data = dict(
                object_fg_flags=object_fg_flags,
                object_classes=object_classes,
                object_poses=object_poses,
                grasp_pose=np.hstack(ee_to_world),  # in world
                grasp_pose_wrt_obj=np.hstack(ee_to_obj),  # in obj
                reorient_pose=np.hstack(obj_af_to_world),
                solved=solved,
            )

            name = f"class_{'_'.join(str(c) for c in args.class_ids)}"

            while True:
                pkl_file = home / f"data/mercury/reorient/{name}/{i:08d}.pkl"
                if not pkl_file.exists():
                    break
                i += process_num
                if i >= args.size:
                    return

            pkl_file.parent.makedirs_p()
            with open(pkl_file, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Saved to: {pkl_file}")


if __name__ == "__main__":
    main()
