#!/usr/bin/env python

import argparse
import itertools
import time

from loguru import logger
import numpy as np
import open3d
import pybullet_planning as pp
import sklearn.neighbors
from yarr.agents.agent import ActResult

import mercury

import _open3d
import common_utils
from env import PickAndPlaceEnv


def visualize(
    env,
    query_ocs,
    query_ocs_normal_ends,
    target_point,
    target_point_normal_end,
    T_obj_af_to_world,
):
    vis = open3d.visualization.Visualizer()
    vis.create_window(height=int(480 * 1.5), width=int(640 * 1.5))

    vis.add_geometry(
        _open3d.LineSet(
            points=[target_point, target_point_normal_end],
            lines=np.array([[0, 1]]),
        )
    )

    # obj -> obj_af
    class_id = common_utils.get_class_id(env.fg_object_id)
    visual_file = mercury.datasets.ycb.get_visual_file(class_id)
    geometry = open3d.io.read_triangle_mesh(visual_file)
    geometry.transform(T_obj_af_to_world)
    vis.add_geometry(geometry)

    # obj -> obj_fn
    T_obj_fn_to_world = mercury.geometry.transformation_matrix(*env.PLACE_POSE)
    geometry = open3d.io.read_triangle_mesh(visual_file)
    geometry.transform(T_obj_fn_to_world)
    vis.add_geometry(geometry)

    vis.add_geometry(_open3d.TriangleMesh.create_coordinate_frame(size=0.2))

    # # target ocs
    # geometry = trimesh.PointCloud(vertices=ocs)
    # geometries.append(_open3d.PointCloud.from_trimesh(geometry))

    # # target ocs aabb
    # aabb = ocs.min(axis=0), ocs.max(axis=0)
    # geometry = trimesh.path.creation.box_outline(extents=aabb[1] - aabb[0])  # NOQA
    # geometry.apply_translation(np.mean(aabb, axis=0))
    # geometries.append(_open3d.LineSet.from_trimesh(geometry))

    # result ocs
    p = np.random.permutation(len(query_ocs))[:1000]
    geometry = _open3d.PointCloud(
        points=query_ocs[p],
        colors=(0, 1, 0),
        normals=(query_ocs_normal_ends - query_ocs)[p],
    )
    vis.add_geometry(geometry)

    for object_id in env.object_ids:
        class_id = common_utils.get_class_id(object_id)
        visual_file = mercury.datasets.ycb.get_visual_file(class_id)
        geometry = open3d.io.read_triangle_mesh(visual_file)
        geometry.transform(
            mercury.geometry.transformation_matrix(*pp.get_pose(object_id))
        )
        vis.add_geometry(geometry)

    view_control = vis.get_view_control()
    camera = pp.get_camera()
    view_control.set_zoom(camera.dist)
    view_control.set_front(-np.asarray(camera.cameraForward))
    view_control.set_lookat(camera.target)
    view_control.set_up(camera.cameraUp)

    vis.run()
    vis.destroy_window()


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


def plan_reorient(env, T_obj_af_to_world, iterations=5):
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

    logger.info("Started planning reorientation")

    indices = np.argwhere(mask)
    env.random_state.shuffle(indices)
    for y, x in indices[:iterations]:
        # lock_renderer = pp.LockRenderer()
        world_saver = pp.WorldSaver()

        result = {}

        def before_return():
            env.ri.attachments = []
            world_saver.restore()
            # lock_renderer.restore()

        object_id = segm[y, x]
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

        result["j_init"] = env.ri.getj()

        # solve j_grasp
        c = mercury.geometry.Coordinate(
            *mercury.geometry.pose_from_matrix(T_ee_af_to_world)
        )
        j = env.ri.solve_ik(c.pose)
        if j is None:
            logger.warning("j_grasp is not found")
            before_return()
            continue
        result["j_grasp"] = j

        env.ri.setj(result["j_grasp"])

        obj_to_world = pp.get_pose(object_id)
        ee_to_world = env.ri.get_pose("tipLink")
        obj_to_ee = pp.multiply(pp.invert(ee_to_world), obj_to_world)
        attachments = [
            pp.Attachment(env.ri.robot, env.ri.ee, obj_to_ee, object_id)
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
            continue
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
            continue
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
            continue
        result["js_pre_grasp"] = js

        env.ri.setj(result["j_grasp"])

        # solve js_place
        env.ri.attachments = attachments
        obstacles = [env.plane, env.bin] + env.object_ids
        obstacles.remove(object_id)
        js = env.ri.planj(
            result["j_place"],
            obstacles=obstacles,
            min_distances={(object_id, -1): -0.01},
        )
        if js is None:
            logger.warning("js_place is not found")
            before_return()
            continue
        result["js_place"] = js

        before_return()
        break

    success = "js_place" in result
    if success:
        logger.success("Found the solution for reorientation")
    else:
        logger.error("Cannot find the solution for reorientation")

    return success, result


def reorient(env, iterations=5):
    query_ocs, query_ocs_normal_ends = get_query_ocs(env)

    obj_to_world = pp.get_pose(env.fg_object_id)
    query_ocs = mercury.geometry.transform_points(
        query_ocs, mercury.geometry.transformation_matrix(*obj_to_world)
    )
    query_ocs_normal_ends = mercury.geometry.transform_points(
        query_ocs_normal_ends,
        mercury.geometry.transformation_matrix(*obj_to_world),
    )

    results = []

    indices = env.random_state.permutation(query_ocs.shape[0])
    for index in indices[:iterations]:
        target_point = query_ocs[index]
        target_point_normal_end = query_ocs_normal_ends[index]

        quaternion = mercury.geometry.quaternion_from_vec2vec(
            v1=target_point_normal_end - target_point,
            v2=[0, 0, -1],
            flip=False,
        )
        T_obj_to_world = mercury.geometry.transformation_matrix(*obj_to_world)
        T_obj_to_obj_af_in_world = mercury.geometry.transform_around(
            mercury.geometry.quaternion_matrix(quaternion),
            pp.get_pose(env.fg_object_id)[0],
        )
        T_obj_af_to_world = T_obj_to_obj_af_in_world @ T_obj_to_world

        aabb = (
            env.PILE_POSITION + (-0.2, -0.2, 0),
            env.PILE_POSITION + (0.2, 0.2, 0.4),
        )

        for dx, dy, dz, dg in itertools.product(
            np.linspace(-0.2, 0.2, 4),
            np.linspace(-0.2, 0.2, 4),
            np.linspace(-0.2, 0.2, 4),
            np.linspace(-np.pi, np.pi, 4),
        ):
            c = mercury.geometry.Coordinate.from_matrix(T_obj_af_to_world)
            c.translate([dx, dy, dz], wrt="world")
            c.rotate([0, 0, dg], wrt="world")

            if not ((c.position > aabb[0]) & (c.position < aabb[1])).all():
                continue

            with pp.LockRenderer(), pp.WorldSaver():
                pp.set_pose(env.fg_object_id, c.pose)
                if mercury.pybullet.is_colliding(env.fg_object_id):
                    continue

            obj_af = mercury.pybullet.duplicate(
                env.fg_object_id,
                collision=True,
                texture=False,
                rgba_color=(0, 1, 0, 0.5),
                position=c.position,
                quaternion=c.quaternion,
            )

            success, result = plan_reorient(env, c.matrix)

            pp.remove_body(obj_af)

            if not success:
                continue

            js_length = 0
            j_prev = None
            for j in result["js_place"]:
                if j_prev is not None:
                    js_length += np.linalg.norm(j_prev - j)
                j_prev = j

            results.append((result, js_length))
            if len(results) >= 10:
                break
        if len(results) >= 10:
            break

    result, _ = min(results, key=lambda x: x[1])

    if 0:
        visualize(
            env,
            query_ocs,
            query_ocs_normal_ends,
            target_point,
            target_point_normal_end,
            c.matrix,
        )

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


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=5, help="seed")
    parser.add_argument("--pause", action="store_true", help="pause")
    args = parser.parse_args()

    env = PickAndPlaceEnv()
    env.random_state = np.random.RandomState(args.seed)
    env.reset(
        pile_file="/home/wkentaro/data/mercury/pile_generation/00001000.npz"
    )

    common_utils.pause(args.pause)

    reorient(env)

    env.setj_to_camera_pose()
    env.update_obs()

    ocs = env.obs["ocs"].transpose(1, 2, 0)
    fg_mask = env.obs["fg_mask"]

    query_ocs, _ = get_query_ocs(env)
    kdtree = sklearn.neighbors.KDTree(ocs.reshape(-1, 3))
    distances, indices = kdtree.query(query_ocs)
    a_flattens = indices[
        (fg_mask.flatten()[indices] == 1) & (distances < 0.001)
    ]

    a_flattens = np.unique(a_flattens)
    env.random_state.shuffle(a_flattens)

    for a_flatten in a_flattens:
        action = (a_flatten // ocs.shape[1], a_flatten % ocs.shape[1])
        act_result = ActResult(action)
        is_valid, validation_result = env.validate_action(act_result)
        if is_valid:
            act_result.validation_result = validation_result
            break
    else:
        logger.error("No valid actions")
        return

    env.step(act_result)


if __name__ == "__main__":
    main()
