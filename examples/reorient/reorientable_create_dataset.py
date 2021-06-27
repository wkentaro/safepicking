#!/usr/bin/env python

import argparse
import itertools
import pickle

from loguru import logger
import numpy as np
import path
import pybullet_planning as pp

import mercury

from _env import Env
import _reorient
import _utils
from pickable_reorient_poses import get_reorient_poses


home = path.Path("~").expanduser()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, required=True, help="seed")
    parser.add_argument("--gui", action="store_true", help="gui")
    parser.add_argument("--visualize", action="store_true", help="visualize")
    args = parser.parse_args()

    if (
        home
        / "data/mercury/reorient/reorientable"
        / f"s-{args.seed:08d}/00000099.pkl"
    ).exists():
        return

    env = Env(class_ids=[2, 3, 5, 11, 12, 15], gui=args.gui)
    env.random_state = np.random.RandomState(args.seed)
    env.launch()

    with pp.LockRenderer():
        env.reset()
        for obj in mercury.pybullet.get_body_unique_ids():
            if obj in [env.plane, env.ri.robot] + env.object_ids:
                continue
            pp.remove_body(obj)

    object_fg_flags = []
    object_classes = []
    object_poses = []
    for object_id in env.object_ids:
        object_fg_flags.append(object_id == env.fg_object_id)
        object_classes.append(_utils.get_class_id(object_id))
        object_poses.append(np.hstack(pp.get_pose(object_id)))
    object_fg_flags = np.array(object_fg_flags, dtype=bool)
    object_classes = np.array(object_classes, dtype=np.int32)
    object_poses = np.array(object_poses, dtype=np.float32)

    reorient_poses = get_reorient_poses(env)

    if args.visualize:
        for reorient_pose in reorient_poses:
            mercury.pybullet.duplicate(
                env.fg_object_id,
                position=reorient_pose[:3],
                quaternion=reorient_pose[3:],
                collision=False,
            )
        return

    grasp_poses = np.array(
        list(itertools.islice(_reorient.get_grasp_poses(env), 100))
    )

    n_saved = 0
    for reorient_pose in reorient_poses[
        np.random.permutation(reorient_poses.shape[0])
    ]:
        index = np.random.permutation(grasp_poses.shape[0])[0]
        grasp_pose = grasp_poses[index]

        result = _reorient.plan_reorient(env, grasp_pose, reorient_pose)

        graspable = "j_grasp" in result
        placable = "j_place" in result
        reorientable = "js_place" in result
        trajectory_length = result.get("js_place_length", np.nan)

        if args.gui and reorientable:
            with pp.WorldSaver():
                _reorient.execute_reorient(env, result)

        if args.gui:
            continue

        ee_to_world = np.hsplit(grasp_pose, [3])
        obj_to_world = np.hsplit(object_poses[object_fg_flags][0], [3])
        world_to_obj = pp.invert(obj_to_world)
        ee_to_obj = pp.multiply(world_to_obj, ee_to_world)

        data = dict(
            object_fg_flags=object_fg_flags,
            object_classes=object_classes,
            object_poses=object_poses,
            reorient_pose=reorient_pose,
            grasp_pose=grasp_pose,  # wrt world
            grasp_pose_wrt_obj=np.hstack(ee_to_obj),  # wrt obj
            graspable=graspable,
            placable=placable,
            reorientable=reorientable,
            trajectory_length=trajectory_length,
        )

        while True:
            pkl_file = (
                home
                / "data/mercury/reorient/reorientable"
                / f"s-{args.seed:08d}/{n_saved:08d}.pkl"
            )
            if not pkl_file.exists():
                break
            n_saved += 1

        if n_saved > 99:
            return

        pkl_file.parent.makedirs_p()
        with open(pkl_file, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved to: {pkl_file}")

        n_saved += 1


if __name__ == "__main__":
    main()
