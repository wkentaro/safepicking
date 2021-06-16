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

from env import Env


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
    env.random_state.shuffle(indices)

    obstacles = [env.plane, env.bin] + env.object_ids
    obstacles.remove(env.fg_object_id)

    for y, x in indices:
        position = pcd_in_ee[y, x]
        quaternion = mercury.geometry.quaternion_from_vec2vec(
            [0, 0, 1], normals_in_ee[y, x]
        )
        ee_af_to_ee = position, quaternion
        ee_af_to_world = pp.multiply(ee_to_world, ee_af_to_ee)

        j = env.ri.solve_ik(ee_af_to_world)
        if j is not None and env.ri.validatej(j, obstacles=obstacles):
            yield np.hstack(ee_af_to_world)


def plan_reorient(env, c_grasp, c_reorient):
    T_ee_af_to_world = c_grasp.matrix
    T_obj_af_to_world = c_reorient.matrix

    # lock_renderer = pp.LockRenderer()
    world_saver = pp.WorldSaver()

    result = {}

    result["grasp_pose"] = np.hstack(c_grasp.pose)
    result["reorient_pose"] = np.hstack(c_reorient.pose)

    def before_return():
        env.ri.attachments = []
        world_saver.restore()
        # lock_renderer.restore()

    result["j_init"] = env.ri.getj()

    bg_object_ids = [env.plane, env.bin] + env.object_ids
    bg_object_ids.remove(env.fg_object_id)

    # solve j_grasp
    c = mercury.geometry.Coordinate(
        *mercury.geometry.pose_from_matrix(T_ee_af_to_world)
    )
    j = env.ri.solve_ik(c.pose)
    if j is None:
        logger.warning("j_grasp is not found")
        before_return()
        return result
    if not env.ri.validatej(j, obstacles=bg_object_ids):
        logger.warning("j_grasp is invalid")
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
            thre=0.01,
            rthre=np.deg2rad(10),
        )
    if j is None:
        logger.warning("j_place is not found")
        before_return()
        return result
    if not env.ri.validatej(
        j,
        obstacles=bg_object_ids,
        min_distances={(env.ri.attachments[0].child, -1): -0.02},
    ):
        logger.warning("j_place is invalid")
        before_return()
        return result
    result["j_place"] = j
    env.ri.attachments = []

    # solve js_grasp
    js = []
    for _ in range(10):
        c.translate([0, 0, -0.01])
        j = env.ri.solve_ik(c.pose, n_init=1)
        if j is None:
            break
        js.append(j)
    js = js[::-1]
    if j is None:
        logger.warning("js_grasp is not found")
        before_return()
        return result
    result["js_grasp"] = js

    if not env.ri.validatej(
        js[0], obstacles=bg_object_ids + [env.fg_object_id]
    ):
        logger.warning("j_pre_grasp is invalid")
        before_return()
        return result
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
    env.ri.attachments = attachments
    env.ri.attachments[0].assign()

    # solve js_place
    obstacles = [env.plane, env.bin] + env.object_ids
    obstacles.remove(env.ri.attachments[0].child)
    js = env.ri.planj(
        result["j_place"],
        obstacles=obstacles,
        min_distances_start_goal={(env.ri.attachments[0].child, -1): -0.01},
    )
    if js is None:
        logger.warning("js_place is not found")
        before_return()
        return result
    result["js_place"] = js

    result["js_place_length"] = 0
    j_prev = result["j_grasp"]
    for j in result["js_place"]:
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

    for _ in env.ri.grasp(
        min_dz=0.08, max_dz=0.12, rotation_axis=True, speed=0.001
    ):
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

    c = mercury.geometry.Coordinate(*env.ri.get_pose("tipLink"))
    c.translate([0, 0, -0.1])
    c.translate([0, 0, 0.2], wrt="world")
    j = env.ri.solve_ik(c.pose, rotation_axis=False)
    if j is not None:
        js = env.ri.planj(
            j,
            obstacles=[env.plane, env.bin] + env.object_ids,
            min_distances_start_goal=mercury.utils.StaticDict(-0.02),
        )
        if js is not None:
            for _ in (_ for j in js for _ in env.ri.movej(j, speed=0.005)):
                pp.step_simulation()
                time.sleep(pp.get_time_step())

    for _ in env.ri.move_to_homej(
        bg_object_ids=[env.plane, env.bin],
        object_ids=env.object_ids,
    ):
        pp.step_simulation()
        time.sleep(pp.get_time_step())


def rollout_plan_reorient(
    env,
    return_failed=False,
    grasp_num_sample=4,
    min_angle=np.deg2rad(0),
    max_angle=np.deg2rad(10),
):
    from reorient_poses import get_reorient_poses2

    reorient_poses, angles, _, _ = get_reorient_poses2(env)
    keep = (min_angle <= angles) & (angles < max_angle)
    reorient_poses = reorient_poses[keep]

    grasp_poses = np.array(list(itertools.islice(get_grasp_poses(env), 32)))

    i = 0
    for reorient_pose in reorient_poses:
        c_reorient = mercury.geometry.Coordinate(
            reorient_pose[:3], reorient_pose[3:]
        )
        for grasp_pose in grasp_poses[np.random.permutation(len(grasp_poses))][
            :grasp_num_sample
        ]:
            c_grasp = mercury.geometry.Coordinate(
                grasp_pose[:3], grasp_pose[3:]
            )

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

    obs = env.obs

    normal = -(point_normal_end - point)  # flip
    eye = point + 0.5 * normal

    T_cam_to_world = mercury.geometry.look_at(eye, target)
    fovy = np.deg2rad(60)
    height = 240
    width = 240
    rgb, depth, segm = mercury.pybullet.get_camera_image(
        T_cam_to_world,
        fovy,
        height,
        width,
    )

    fg_mask = segm == env.fg_object_id
    K = env.ri.get_opengl_intrinsic_matrix()
    pcd_in_camera = mercury.geometry.pointcloud_from_depth(
        depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )
    pcd_in_world = mercury.geometry.transform_points(
        pcd_in_camera, T_cam_to_world
    )
    ocs = np.zeros_like(pcd_in_world)
    for obj in env.object_ids:
        world_to_obj = pp.invert(pp.get_pose(obj))
        ocs[segm == obj] = mercury.geometry.transform_points(
            pcd_in_world,
            mercury.geometry.transformation_matrix(*world_to_obj),
        )[segm == obj]

    env.obs = dict(
        rgb=rgb.transpose(2, 0, 1),
        depth=depth,
        ocs=ocs.transpose(2, 0, 1).astype(np.float32),
        fg_mask=fg_mask.astype(np.uint8),
        segm=segm,
        camera_to_world=np.hstack(
            mercury.geometry.pose_from_matrix(T_cam_to_world)
        ),
    )

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
            break
    else:
        logger.error("No valid actions")
        env.obs = obs
        return False

    env.execute(validation_result)
    env.obs = obs
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
    parser.add_argument("--on-plane", action="store_true", help="on plane")
    parser.add_argument("--timeout", type=float, default=9, help="timeout")
    args = parser.parse_args()

    env = Env(class_ids=args.class_ids, mp4=args.mp4)
    env.random_state = np.random.RandomState(args.seed)
    env.eval = True
    env.launch()

    with pp.LockRenderer():
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

    for i in itertools.count():
        if plan_and_execute_place(env):
            break

        if i == 2:
            break

        env.setj_to_camera_pose()
        env.update_obs()

        result = {}
        for min_angle, max_angle in [(0, 10), (10, 80), (85, 95)]:
            print(min_angle, max_angle)
            t_start = time.time()
            for result in rollout_plan_reorient(
                env,
                return_failed=True,
                min_angle=np.deg2rad(min_angle),
                max_angle=np.deg2rad(max_angle),
            ):
                if "js_place_length" in result:
                    break
                if (time.time() - t_start) > (args.timeout / 3):
                    break
            if "js_place_length" in result:
                break
        execute_plan(env, result)


if __name__ == "__main__":
    main()
