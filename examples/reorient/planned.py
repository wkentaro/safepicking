#!/usr/bin/env python

import argparse
import itertools
import time

from loguru import logger
import numpy as np
import pybullet_planning as pp
import sklearn.neighbors
from yarr.agents.agent import ActResult

import mercury

from pick_and_place_env import PickAndPlaceEnv


def get_query_ocs(env):
    # get query ocs
    with pp.LockRenderer(), pp.WorldSaver():
        pp.set_pose(env.fg_object_id, env.PLACE_POSE)

        T_camera_to_world = mercury.geometry.look_at(
            env.BIN_POSE[0] + [0, -0.3, 0],
            env.BIN_POSE[0],
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


def get_reorient_poses(
    env, num_delta=4, num_sample=4, centroid=False, discretize=True
):
    query_ocs, query_ocs_normal_ends = get_query_ocs(env)

    obj_to_world = pp.get_pose(env.fg_object_id)
    T_obj_to_world = mercury.geometry.transformation_matrix(*obj_to_world)
    query_ocs = mercury.geometry.transform_points(query_ocs, T_obj_to_world)
    query_ocs_normal_ends = mercury.geometry.transform_points(
        query_ocs_normal_ends, T_obj_to_world
    )

    if centroid:
        assert num_sample == 1
        query_ocs_centroid = query_ocs.mean(axis=0)
        distances = np.linalg.norm(query_ocs - query_ocs_centroid, axis=1)
        indices = [np.argmin(distances)]
    else:
        indices = env.random_state.permutation(query_ocs.shape[0])[:num_sample]

    for i in indices:
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

        if discretize:
            deltas = np.array(
                list(
                    itertools.product(
                        np.linspace(-0.1, 0.1, num_delta),
                        np.linspace(-0.1, 0.1, num_delta),
                        np.linspace(-0.1, 0.1, num_delta),
                        np.linspace(-np.pi, np.pi, num_delta),
                    )
                )
            )
        else:
            deltas = env.random_state.uniform(
                [-0.1, -0.1, -0.1, -np.pi],
                [0.1, 0.1, 0.1, np.pi],
                size=(num_delta ** 4, 4),
            )
        for dx, dy, dz, dg in deltas[
            env.random_state.permutation(deltas.shape[0])
        ]:
            c = mercury.geometry.Coordinate.from_matrix(T_obj_af_to_world)
            c.translate([dx, dy, dz], wrt="world")
            c.rotate([0, 0, dg], wrt="world")

            with pp.LockRenderer(), pp.WorldSaver():
                pp.set_pose(env.fg_object_id, c.pose)
                if mercury.pybullet.is_colliding(
                    env.fg_object_id, distance=-0.01
                ):
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
        return result
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
        return result
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
        return result
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
        return result
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
        return result
    result["js_place"] = js

    result["js_place_length"] = 0
    j_prev = None
    for j in result["js_place"]:
        if j_prev is not None:
            result["js_place_length"] += np.linalg.norm(j_prev - j)
        j_prev = j

    before_return()

    if "js_place_length" in result:
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
    for _ in (_ for j in js for _ in env.ri.movej(j, speed=0.005)):
        pp.step_simulation()
        time.sleep(pp.get_time_step())

    for _ in range(int(1 / pp.get_time_step())):
        pp.step_simulation()
        time.sleep(pp.get_time_step())

    env.ri.ungrasp()

    for _ in range(int(3 / pp.get_time_step())):
        pp.step_simulation()
        time.sleep(pp.get_time_step())


def rollout_plan_reorient(
    env,
    return_failed=False,
    grasp_num_sample=4,
    threshold=np.deg2rad(95),
):
    from reorient_poses import get_reorient_poses2

    i = 0
    for c_reorient in get_reorient_poses2(env, threshold=threshold)[0]:
        c_reorient = mercury.geometry.Coordinate(*c_reorient)
        for c_grasp in itertools.islice(
            get_grasp_poses(env), grasp_num_sample
        ):
            debug = pp.add_text(
                f"plan {i:04d}",
                position=c_reorient.position + [0, 0, 0.1],
            )
            i += 1

            obj_af = mercury.pybullet.duplicate(
                env.fg_object_id,
                collision=False,
                rgba_color=(0, 1, 0, 0.5),
                position=c_reorient.position,
                quaternion=c_reorient.quaternion,
            )

            result = plan_reorient(env, c_grasp=c_grasp, c_reorient=c_reorient)
            success = "js_place_length" in result

            pp.remove_body(obj_af)
            pp.remove_debug(debug)

            result["c_init"] = mercury.geometry.Coordinate(
                *pp.get_pose(env.fg_object_id)
            )
            result["c_grasp"] = c_grasp
            result["c_reorient"] = c_reorient

            if return_failed or success:
                yield result


def plan_and_execute_place(env, num_sample=5):
    query_ocs, query_ocs_normal_ends = get_query_ocs(env)

    i = np.argmin(np.linalg.norm(query_ocs - query_ocs.mean(axis=0), axis=1))
    point = query_ocs[i]
    point_normal_end = query_ocs_normal_ends[i]

    obj_to_world = pp.get_pose(env.fg_object_id)
    point, point_normal_end = mercury.geometry.transform_points(
        [point, point_normal_end],
        mercury.geometry.transformation_matrix(*obj_to_world),
    )
    target = point

    normal = -(point_normal_end - point)  # flip
    for normal in np.linspace(normal, [0, 0, 1], num=10):
        eye = point + 0.5 * normal
        # pp.add_line(eye, target)
        cam_to_world = mercury.geometry.pose_from_matrix(
            mercury.geometry.look_at(eye, target)
        )
        # pp.draw_pose(cam_to_world)
        j = env.ri.solve_ik(
            cam_to_world,
            move_target=env.ri.robot_model.camera_link,
            rotation_axis="z",
        )
        if j is not None:
            env.ri.setj(j)
            break
    else:
        env.setj_to_camera_pose()
    env.update_obs()

    ocs = env.obs["ocs"].transpose(1, 2, 0)
    fg_mask = env.obs["fg_mask"]

    query_ocs, _ = get_query_ocs(env)
    kdtree = sklearn.neighbors.KDTree(ocs.reshape(-1, 3))
    distances, indices = kdtree.query(query_ocs)
    a_flattens = indices[
        (fg_mask.flatten()[indices] == 1) & (distances < 0.01)
    ]

    a_flattens = np.unique(a_flattens)
    env.random_state.shuffle(a_flattens)

    if 0:
        obj_to_world = pp.get_pose(env.fg_object_id)
        for a_flatten in a_flattens:
            action = (a_flatten // ocs.shape[1], a_flatten % ocs.shape[1])
            point = ocs[action[0], action[1]]
            point = mercury.geometry.transform_points(
                [point], mercury.geometry.transformation_matrix(*obj_to_world)
            )[0]
            pp.draw_point(point, color=(0, 1, 0, 1))

    for a_flatten in a_flattens[:num_sample]:
        action = (a_flatten // ocs.shape[1], a_flatten % ocs.shape[1])
        act_result = ActResult(action)
        is_valid, validation_result = env.validate_action(act_result)
        if is_valid:
            act_result.validation_result = validation_result
            break
    else:
        logger.error("No valid actions")
        return False

    env.step(act_result)
    return True


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--class-ids", type=int, nargs="+", help="class ids", required=True
    )
    parser.add_argument("--mp4", help="mp4")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--timeout", type=float, default=3, help="timeout")
    parser.add_argument("--on-plane", action="store_true", help="on plane")
    args = parser.parse_args()

    env = PickAndPlaceEnv(class_ids=args.class_ids, mp4=args.mp4)
    env.random_state = np.random.RandomState(args.seed)
    env.eval = True
    env.reset()

    if args.on_plane:
        with pp.LockRenderer():
            for object_id in env.object_ids:
                if object_id != env.fg_object_id:
                    pp.remove_body(object_id)

            for _ in range(2400):
                pp.step_simulation()
        env.object_ids = [env.fg_object_id]
        env.update_obs()

    t_start = time.time()
    results = []
    for result in rollout_plan_reorient(env, return_failed=True):
        if "js_place_length" in result:
            results.append(result)
        if (time.time() - t_start) > args.timeout:
            break

    result = min(results, key=lambda x: x["js_place_length"])
    execute_plan(env, result)


if __name__ == "__main__":
    main()
