#!/usr/bin/env python

import itertools
import time

from loguru import logger
import numpy as np
import pybullet_planning as pp

import mercury

from pick_and_place_env import PickAndPlaceEnv


def get_query_ocs(env):
    # get query ocs
    with pp.LockRenderer(), pp.WorldSaver():
        pp.set_pose(env.fg_object_id, env.PLACE_POSE)

        T_camera_to_world = mercury.geometry.look_at(
            [0, 0.3, 0.6], [0, 0.6, 0.6]
        )
        fovy = np.deg2rad(60)
        height = 240
        width = 240
        mercury.pybullet.draw_camera(
            fovy,
            height,
            width,
            pose=mercury.geometry.pose_from_matrix(T_camera_to_world),
        )
        rgb, depth, segm = mercury.pybullet.get_camera_image(
            T_camera_to_world, fovy, height, width
        )
        K = mercury.geometry.opengl_intrinsic_matrix(fovy, height, width)
        pcd_in_camera = mercury.geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )
        pcd_in_world = mercury.geometry.transform_points(
            pcd_in_camera, T_camera_to_world
        )
        normals_in_world = mercury.geometry.normals_from_pointcloud(
            pcd_in_world
        )
        pcd_normal_ends_in_world = pcd_in_world + normals_in_world

        world_to_obj = pp.invert(pp.get_pose(env.fg_object_id))
        mask = segm == env.fg_object_id
        query_ocs = mercury.geometry.transform_points(
            pcd_in_world[mask],
            mercury.geometry.transformation_matrix(*world_to_obj),
        )
        query_ocs_normal_ends = mercury.geometry.transform_points(
            pcd_normal_ends_in_world[mask],
            mercury.geometry.transformation_matrix(*world_to_obj),
        )

    return query_ocs, query_ocs_normal_ends


def get_reorient_poses(env):
    query_ocs, query_ocs_normal_ends = get_query_ocs(env)

    obj_to_world = pp.get_pose(env.fg_object_id)
    T_obj_to_world = mercury.geometry.transformation_matrix(*obj_to_world)
    query_ocs = mercury.geometry.transform_points(query_ocs, T_obj_to_world)
    query_ocs_normal_ends = mercury.geometry.transform_points(
        query_ocs_normal_ends, T_obj_to_world
    )

    bounds = (
        env.PILE_POSITION + (-0.2, -0.2, 0),
        env.PILE_POSITION + (0.2, 0.2, 0.4),
    )

    for i in env.random_state.permutation(query_ocs.shape[0]):
        quaternion = mercury.geometry.quaternion_from_vec2vec(
            v1=query_ocs_normal_ends[i] - query_ocs[i],
            v2=[0, 0, -1],
            flip=False,
        )
        T_obj_to_obj_af_in_world = mercury.geometry.transform_around(
            mercury.geometry.quaternion_matrix(quaternion),
            mercury.geometry.translation_from_matrix(T_obj_to_world),
        )
        T_obj_af_to_world = T_obj_to_obj_af_in_world @ T_obj_to_world

        deltas = np.array(
            list(
                itertools.product(
                    np.linspace(-0.2, 0.2, 4),
                    np.linspace(-0.2, 0.2, 4),
                    np.linspace(-0.2, 0.2, 4),
                    np.linspace(-np.pi, np.pi, 4),
                )
            )
        )
        for dx, dy, dz, dg in deltas[
            env.random_state.permutation(deltas.shape[0])
        ]:
            c = mercury.geometry.Coordinate.from_matrix(T_obj_af_to_world)
            c.translate([dx, dy, dz], wrt="world")
            c.rotate([0, 0, dg], wrt="world")

            if not ((c.position > bounds[0]) & (c.position < bounds[1])).all():
                continue

            with pp.LockRenderer(), pp.WorldSaver():
                pp.set_pose(env.fg_object_id, c.pose)
                if mercury.pybullet.is_colliding(env.fg_object_id):
                    continue

            yield c


def get_grasp_poses(env):
    segm = env.obs["segm"]
    depth = env.obs["depth"]

    K = env.ri.get_opengl_intrinsic_matrix()
    mask = segm == env.fg_object_id
    pcd_in_camera = mercury.geometry.pointcloud_from_depth(
        depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )
    camera_to_world = env.ri.get_pose("camera_link")
    ee_to_world = env.ri.get_pose("tipLink")
    camera_to_ee = pp.multiply(pp.invert(ee_to_world), camera_to_world)
    pcd_in_ee = mercury.geometry.transform_points(
        pcd_in_camera, mercury.geometry.transformation_matrix(*camera_to_ee)
    )
    normals_in_ee = mercury.geometry.normals_from_pointcloud(pcd_in_ee)

    indices = np.argwhere(mask)
    for y, x in indices[env.random_state.permutation(indices.shape[0])]:
        position = pcd_in_ee[y, x]
        quaternion = mercury.geometry.quaternion_from_vec2vec(
            [0, 0, 1], normals_in_ee[y, x]
        )
        T_ee_to_ee_af_in_ee = mercury.geometry.transformation_matrix(
            position, quaternion
        )

        T_ee_to_world = mercury.geometry.transformation_matrix(
            *env.ri.get_pose("tipLink")
        )
        T_ee_to_ee = np.eye(4)
        T_ee_af_to_ee = T_ee_to_ee_af_in_ee @ T_ee_to_ee
        T_ee_af_to_world = T_ee_to_world @ T_ee_af_to_ee

        c = mercury.geometry.Coordinate.from_matrix(T_ee_af_to_world)
        yield c


def plan_reorient(env, c_grasp, c_reorient):
    T_ee_af_to_world = c_grasp.matrix
    T_obj_af_to_world = c_reorient.matrix

    # lock_renderer = pp.LockRenderer()
    world_saver = pp.WorldSaver()

    result = {}

    def before_return():
        env.ri.attachments = []
        world_saver.restore()
        # lock_renderer.restore()

    result["j_init"] = env.ri.getj()

    # solve j_grasp
    c = mercury.geometry.Coordinate(
        *mercury.geometry.pose_from_matrix(T_ee_af_to_world)
    )
    j = env.ri.solve_ik(c.pose)
    if j is None:
        logger.warning("j_grasp is not found")
        before_return()
        return False, result
    result["j_grasp"] = j

    env.ri.setj(result["j_grasp"])

    obj_to_world = pp.get_pose(env.fg_object_id)
    ee_to_world = env.ri.get_pose("tipLink")
    obj_to_ee = pp.multiply(pp.invert(ee_to_world), obj_to_world)
    attachments = [
        pp.Attachment(env.ri.robot, env.ri.ee, obj_to_ee, env.fg_object_id)
    ]

    # solve j_place
    env.ri.attachments = attachments
    with env.ri.enabling_attachments():
        j = env.ri.solve_ik(
            mercury.geometry.pose_from_matrix(T_obj_af_to_world),
            move_target=env.ri.robot_model.attachment_link0,
        )
    if j is None:
        logger.warning("j_place is not found")
        before_return()
        return False, result
    result["j_place"] = j
    env.ri.attachments = []

    # solve js_grasp
    js = []
    for _ in range(10):
        c.translate([0, 0, -0.01])
        j = env.ri.solve_ik(c.pose)
        if j is None:
            break
        js.append(j)
    js = js[::-1]
    if j is None:
        logger.warning("js_grasp is not found")
        before_return()
        return False, result
    result["js_grasp"] = js
    result["j_pre_grasp"] = js[0]

    # solve js_pre_grasp
    env.ri.setj(result["j_init"])
    js = env.ri.planj(
        result["j_pre_grasp"],
        obstacles=[env.plane, env.bin] + env.object_ids,
    )
    if js is None:
        logger.warning("js_pre_grasp is not found")
        before_return()
        return False, result
    result["js_pre_grasp"] = js

    env.ri.setj(result["j_grasp"])

    # solve js_place
    env.ri.attachments = attachments
    obstacles = [env.plane, env.bin] + env.object_ids
    obstacles.remove(env.ri.attachments[0].child)
    js = env.ri.planj(
        result["j_place"],
        obstacles=obstacles,
        min_distances={(env.ri.attachments[0].child, -1): -0.01},
    )
    if js is None:
        logger.warning("js_place is not found")
        before_return()
        return False, result
    result["js_place"] = js

    before_return()

    if "js_place" in result:
        logger.success("Found the solution for reorientation")
    else:
        logger.error("Cannot find the solution for reorientation")

    return result


def execute_plan(env, result):
    js = result["js_pre_grasp"]
    for _ in (_ for j in js for _ in env.ri.movej(j)):
        pp.step_simulation()
        time.sleep(pp.get_time_step())

    for _ in env.ri.grasp(min_dz=0.08, max_dz=0.12, rotation_axis=True):
        pp.step_simulation()
        time.sleep(pp.get_time_step())

    js = result["js_place"]
    for _ in (_ for j in js for _ in env.ri.movej(j)):
        pp.step_simulation()
        time.sleep(pp.get_time_step())

    for _ in range(int(1 / pp.get_time_step())):
        pp.step_simulation()
        time.sleep(pp.get_time_step())

    env.ri.ungrasp()

    for _ in range(int(3 / pp.get_time_step())):
        pp.step_simulation()
        time.sleep(pp.get_time_step())


def rollout_plan_reorient(env, return_failed=False):
    for c_reorient in get_reorient_poses(env):
        for c_grasp in itertools.islice(get_grasp_poses(env), 3):
            obj_af = mercury.pybullet.duplicate(
                env.fg_object_id,
                collision=False,
                texture=False,
                rgba_color=(0, 1, 0, 0.5),
                position=c_reorient.position,
                quaternion=c_reorient.quaternion,
            )

            result = plan_reorient(env, c_grasp=c_grasp, c_reorient=c_reorient)
            success = "js_place" in result

            pp.remove_body(obj_af)

            result["c_grasp"] = c_grasp
            result["c_reorient"] = c_reorient
            result["js_place_length"] = 0

            if success:
                j_prev = None
                for j in result["js_place"]:
                    if j_prev is not None:
                        result["js_place_length"] += np.linalg.norm(j_prev - j)
                    j_prev = j

            if return_failed or success:
                yield result


def main():
    env = PickAndPlaceEnv()
    env.random_state = np.random.RandomState(5)
    env.reset(pile_file=env.PILES_DIR / "00001000.npz")

    results = itertools.islice(rollout_plan_reorient(env), 10)

    result, _ = min(results, key=lambda x: x["js_place_length"])
    execute_plan(env, result)


if __name__ == "__main__":
    main()
