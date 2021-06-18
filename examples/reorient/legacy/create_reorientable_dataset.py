#!/usr/bin/env python

import argparse
import itertools
import pickle
import time

from loguru import logger
import numpy as np
import path
import pybullet_planning as pp

import mercury

import common_utils
from env import Env
from planned import get_grasp_poses
from planned import plan_reorient


home = path.Path("~").expanduser()


def get_reorient_poses(env):
    position = np.array(pp.get_pose(env.fg_object_id)[0])
    aabb = position - 0.3, position + 0.3
    pp.draw_aabb(aabb)

    XY = np.random.uniform(-0.3, 0.3, (10, 2))
    ABG = np.r_[
        np.array(
            list(
                itertools.product(
                    np.linspace(-np.pi, np.pi, num=6, endpoint=False),
                    np.linspace(-np.pi, np.pi, num=6, endpoint=False),
                    np.linspace(-np.pi, np.pi, num=6, endpoint=False),
                )
            )
        ),
        np.random.uniform(-np.pi, np.pi, (84, 3)),
    ]

    reorient_poses = []
    with pp.LockRenderer(), pp.WorldSaver():
        for dx, dy in XY:
            for a, b, g in ABG[np.random.permutation(ABG.shape[0])][:10]:
                x, y = position[:2] + [dx, dy]
                c = mercury.geometry.Coordinate(position=(x, y, 0))
                c.quaternion = mercury.geometry.quaternion_from_euler(
                    (a, b, g)
                )
                pp.set_pose(env.fg_object_id, c.pose)

                c.position[2] = -pp.get_aabb(env.fg_object_id)[0][2]
                pp.set_pose(env.fg_object_id, c.pose)

                points = pp.body_collision_info(
                    env.fg_object_id, env.plane, max_distance=0.2
                )
                distance_to_plane = min(point[8] for point in points)
                assert distance_to_plane > 0
                c.position[2] -= distance_to_plane

                c.position[2] += np.random.uniform(0.02, 0.1)
                pp.set_pose(env.fg_object_id, c.pose)

                if not mercury.pybullet.is_colliding(env.fg_object_id):
                    reorient_poses.append(np.hstack(c.pose))
    return np.array(reorient_poses)


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

    env = Env(class_ids=args.class_ids, gui=args.gui)

    i = process_index
    while True:
        env.reset()

        t_start = time.time()
        grasp_poses = np.array(
            list(itertools.islice(get_grasp_poses(env), 100))
        )
        print(time.time() - t_start)

        t_start = time.time()
        reorient_poses = get_reorient_poses(env)
        print(time.time() - t_start)

        print(len(reorient_poses))
        for reorient_pose in reorient_poses:
            for grasp_pose in grasp_poses[
                np.random.permutation(len(grasp_poses))
            ][:1]:
                result = plan_reorient(
                    env,
                    mercury.geometry.Coordinate(
                        grasp_pose[:3], grasp_pose[3:]
                    ),
                    mercury.geometry.Coordinate(
                        reorient_pose[:3], reorient_pose[3:]
                    ),
                )

                keys = [
                    "j_grasp",
                    "j_place",
                    "js_grasp",
                    "js_pre_grasp",
                    "js_place",
                ]
                solved = np.array([key in result for key in keys], dtype=bool)

                length = result.get("js_place_length", np.nan)

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

                ee_to_world = (
                    result["grasp_pose"][:3],
                    result["grasp_pose"][3:],
                )
                obj_to_world = object_poses[object_fg_flags][0]
                obj_to_world = obj_to_world[:3], obj_to_world[3:]
                ee_to_obj = pp.multiply(pp.invert(obj_to_world), ee_to_world)

                obj_af_to_world = (
                    result["reorient_pose"][:3],
                    result["reorient_pose"][3:],
                )

                data = dict(
                    object_fg_flags=object_fg_flags,
                    object_classes=object_classes,
                    object_poses=object_poses,
                    grasp_pose=np.hstack(ee_to_world),  # in world
                    grasp_pose_wrt_obj=np.hstack(ee_to_obj),  # in obj
                    reorient_pose=np.hstack(obj_af_to_world),
                    solved=solved,
                    length=length,
                )

                if args.gui and "js_place" in result:
                    with pp.WorldSaver():
                        for j in result["js_place"]:
                            env.ri.setj(j)
                            pp.set_pose(
                                env.fg_object_id,
                                pp.multiply(
                                    env.ri.get_pose("tipLink"),
                                    pp.invert(ee_to_obj),
                                ),
                            )
                            time.sleep(0.02)

                if not args.gui:
                    name = f"reorientable-{'_'.join(str(c) for c in args.class_ids)}"  # NOQA

                    while True:
                        pkl_file = (
                            home / f"data/mercury/reorient/{name}/{i:08d}.pkl"
                        )
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
