#!/usr/bin/env python

import time

import numpy as np
import open3d
import pybullet_planning as pp
import sklearn.neighbors
import trimesh
from yarr.agents.agent import ActResult

import mercury

import _open3d
import common_utils
from env import GraspWithIntentEnv


def get_query_ocs(env):
    # get query ocs
    with pp.LockRenderer(), pp.WorldSaver():
        pp.set_pose(env.fg_object_id, [(0, 0.6, 0.6), (0, 0, 0, 1)])

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

        if 0:
            mask = ~np.isnan(depth) & (depth < 1)
            p = np.random.permutation(mask.sum())[:1000]
            geometry = _open3d.PointCloud(
                points=pcd_in_world[mask][p],
                colors=(0, 1.0, 0),
                normals=normals_in_world[mask][p],
            )
            open3d.visualization.draw_geometries([geometry])

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


def reorient(env):
    query_ocs, query_ocs_normal_ends = get_query_ocs(env)

    obj_to_world = pp.get_pose(env.fg_object_id)
    query_ocs = mercury.geometry.transform_points(
        query_ocs, mercury.geometry.transformation_matrix(*obj_to_world)
    )
    query_ocs_normal_ends = mercury.geometry.transform_points(
        query_ocs_normal_ends,
        mercury.geometry.transformation_matrix(*obj_to_world),
    )

    centroid = query_ocs.mean(axis=0)
    nearest_index = np.argmin(np.linalg.norm(query_ocs - centroid, axis=1))
    target_point = query_ocs[nearest_index]
    target_point_normal_end = query_ocs_normal_ends[nearest_index]

    quaternion = mercury.geometry.quaternion_from_vec2vec(
        v1=target_point_normal_end - target_point, v2=[0, 0, -1], flip=False
    )
    T_obj_to_world = mercury.geometry.transformation_matrix(*obj_to_world)
    T_obj_to_obj_af_in_world = mercury.geometry.transform_around(
        mercury.geometry.quaternion_matrix(quaternion),
        pp.get_pose(env.fg_object_id)[0],
    )
    T_obj_af_to_world = T_obj_to_obj_af_in_world @ T_obj_to_world

    if 0:
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

        vis.add_geometry(
            _open3d.TriangleMesh.create_coordinate_frame(size=0.2)
        )

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
            geometry = _open3d.TriangleMesh.from_trimesh(
                trimesh.load_mesh(visual_file, process=False)
            )
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

    for _ in env.ri.random_grasp(
        env.obs["depth"],
        env.obs["segm"],
        bg_object_ids=[env.plane],
        object_ids=env.object_ids,
        target_object_ids=[env.fg_object_id],
        max_angle=np.inf,
        random_state=env.random_state,
    ):
        pp.step_simulation()
        time.sleep(pp.get_time_step() / 10)

    c = mercury.geometry.Coordinate(*env.ri.get_pose("tipLink"))
    c.translate([0, 0, 0.2], wrt="world")
    j = env.ri.solve_ik(c.pose)
    for _ in env.ri.movej(j):
        pp.step_simulation()
        time.sleep(pp.get_time_step() / 10)

    with env.ri.enabling_attachments():
        while True:
            c = mercury.geometry.Coordinate.from_matrix(T_obj_af_to_world)
            c.position = np.random.uniform(
                [0.3, 0, c.position[2] + 0.2], [0.7, 0, c.position[2] + 0.2]
            )
            j = env.ri.solve_ik(
                c.pose,
                move_target=env.ri.robot_model.attachment_link0,
            )
            if j is None:
                continue
            obstacles = [env.plane] + env.object_ids
            obstacles.remove(env.fg_object_id)
            js = env.ri.planj(
                j,
                obstacles=obstacles,
                min_distances={(env.fg_object_id, -1): -0.01},
            )
            if js is None:
                continue
            break
        for _ in (_ for j in js for _ in env.ri.movej(j)):
            pp.step_simulation()
            time.sleep(pp.get_time_step() / 10)

    c = mercury.geometry.Coordinate(*env.ri.get_pose("tipLink"))
    c.translate([0, 0, -0.1], wrt="world")
    j = env.ri.solve_ik(c.pose)
    for _ in env.ri.movej(j):
        pp.step_simulation()
        time.sleep(pp.get_time_step() / 10)

    for _ in range(int(1 / pp.get_time_step())):
        pp.step_simulation()
        time.sleep(pp.get_time_step() / 10)

    env.ri.ungrasp()

    for _ in range(int(1 / pp.get_time_step())):
        pp.step_simulation()
        time.sleep(pp.get_time_step() / 10)


def main():
    env = GraspWithIntentEnv()
    env.random_state = np.random.RandomState(1)
    env.reset(
        pile_file="/home/wkentaro/data/mercury/pile_generation/00001000.npz"
    )

    reorient(env)

    env.setj_to_camera_pose()
    env.update_obs()

    ocs = env.obs["ocs"].transpose(1, 2, 0)

    query_ocs, _ = get_query_ocs(env)
    kdtree = sklearn.neighbors.KDTree(ocs.reshape(-1, 3))
    distances, indices = kdtree.query(query_ocs)
    a_flattens = indices[distances < 0.001]

    for a_flatten in a_flattens:
        action = (a_flatten // ocs.shape[1], a_flatten % ocs.shape[1])
        act_result = ActResult(action)
        is_valid, validation_result = env.validate_action(act_result)
        if is_valid:
            act_result.validation_result = validation_result
            break

    env.step(act_result)


if __name__ == "__main__":
    main()
