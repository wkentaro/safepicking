#!/usr/bin/env python

import argparse
import itertools
import pickle

from loguru import logger
import numpy as np
import path
import pybullet_planning as pp

import mercury

import common_utils
from env import Env


home = path.Path("~").expanduser()


def get_reorient_poses(env):
    bounds = ((0.1, -0.5, 0.01), (0.8, 0.5, 0.01))
    pp.draw_aabb(bounds)

    XY = np.random.uniform(bounds[0][:2], bounds[1][:2], (100, 2))
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

    aabb = pp.get_aabb(env.fg_object_id)
    max_extent = np.sqrt(((aabb[1] - aabb[0]) ** 2).sum())

    reorient_poses = []
    with pp.LockRenderer(), pp.WorldSaver():
        i = 0
        for x, y in XY:
            box = pp.create_box(w=max_extent, l=max_extent, h=10)
            with pp.LockRenderer():
                pp.set_pose(box, ((x, y, 0), (0, 0, 0, 1)))
            obstacles = [env.ri.robot] + env.object_ids
            obstacles.remove(env.fg_object_id)
            if mercury.pybullet.is_colliding(box, ids2=obstacles):
                pp.remove_body(box)
                continue
            pp.remove_body(box)

            i += 1
            pp.draw_point((x, y, 0.01), color=(0, 1, 0, 1))

            for a, b, g in ABG[np.random.permutation(ABG.shape[0])][:10]:
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

            if i == 10:
                break
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
    env.launch()

    i = process_index
    while True:
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
            object_classes.append(common_utils.get_class_id(object_id))
            object_poses.append(np.hstack(pp.get_pose(object_id)))
        object_fg_flags = np.array(object_fg_flags, dtype=bool)
        object_classes = np.array(object_classes, dtype=np.int32)
        object_poses = np.array(object_poses, dtype=np.float32)

        reorient_poses = get_reorient_poses(env)

        class_id = common_utils.get_class_id(env.fg_object_id)
        pcd_file = mercury.datasets.ycb.get_pcd_file(class_id=class_id)
        pcd_in_obj = np.loadtxt(pcd_file)
        normals_in_obj = np.loadtxt(pcd_file.parent / "normals.xyz")
        normal_ends_in_obj = pcd_in_obj + normals_in_obj

        for reorient_pose in reorient_poses:
            with pp.WorldSaver():
                pp.set_pose(
                    env.fg_object_id, (reorient_pose[:3], reorient_pose[3:])
                )

                for _ in range(480):
                    pp.step_simulation()

                obj_to_world = pp.get_pose(env.fg_object_id)
                ee_to_world = env.ri.get_pose("tipLink")
                obj_to_ee = pp.multiply(pp.invert(ee_to_world), obj_to_world)
                T_ee_to_world = mercury.geometry.transformation_matrix(
                    *ee_to_world
                )
                T_obj_to_ee = mercury.geometry.transformation_matrix(
                    *obj_to_ee
                )
                pcd_in_ee = mercury.geometry.transform_points(
                    pcd_in_obj, T_obj_to_ee
                )
                normal_ends_in_ee = mercury.geometry.transform_points(
                    normal_ends_in_obj, T_obj_to_ee
                )
                normals_in_ee = normal_ends_in_ee - pcd_in_ee

                indices = np.random.permutation(pcd_in_ee.shape[0])
                for index in indices[:10]:
                    position = pcd_in_ee[index]
                    quaternion = mercury.geometry.quaternion_from_vec2vec(
                        [0, 0, 1], normals_in_ee[index], flip=False
                    )

                    T_ee_af_to_ee = mercury.geometry.transformation_matrix(
                        position, quaternion
                    )
                    T_ee_af_to_world = T_ee_to_world @ T_ee_af_to_ee

                    ee_af_to_world = mercury.geometry.pose_from_matrix(
                        T_ee_af_to_world
                    )
                    obj_to_ee_af = pp.multiply(
                        pp.invert(ee_af_to_world), obj_to_world
                    )

                    c = mercury.geometry.Coordinate(*ee_af_to_world)
                    c.translate([0, 0, -0.1])

                    pickable = True

                    j = env.ri.solve_ik(ee_af_to_world, rotation_axis="z")
                    if j is not None:
                        env.ri.setj(j)

                        obstacles = env.bg_objects + env.object_ids
                        obstacles.remove(env.fg_object_id)
                        if not env.ri.validatej(
                            j,
                            obstacles=obstacles,
                            min_distances={(env.fg_object_id, -1): -0.01},
                        ):
                            j = None
                    pickable &= j is not None

                    for _ in range(10):
                        if not pickable:
                            break
                        c.translate([0, 0, 0.01])
                        j = env.ri.solve_ik(ee_af_to_world, rotation_axis=True)
                        if j is not None:
                            env.ri.setj(j)

                            obstacles = env.bg_objects + env.object_ids
                            obstacles.remove(env.fg_object_id)
                            if not env.ri.validatej(j, obstacles=obstacles):
                                j = None
                        pickable &= j is not None

                    data = dict(
                        object_fg_flags=object_fg_flags,
                        object_classes=object_classes,
                        object_poses=object_poses,
                        reorient_pose=reorient_pose,
                        final_pose=np.hstack(obj_to_world),
                        grasp_pose_wrt_obj=np.hstack(pp.invert(obj_to_ee_af)),
                        pickable=pickable,
                    )

                    if not args.gui:
                        name = f"pickable-{'_'.join(str(c) for c in args.class_ids)}"  # NOQA

                        while True:
                            pkl_file = (
                                home
                                / f"data/mercury/reorient/{name}/{i:08d}.pkl"
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
