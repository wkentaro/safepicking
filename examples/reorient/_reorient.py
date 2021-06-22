import itertools
import time

# import imgviz
from loguru import logger
import numpy as np
import pybullet_planning as pp
import sklearn.neighbors

from yarr.agents.agent import ActResult

import mercury


def get_query_ocs(env):
    with pp.LockRenderer(), pp.WorldSaver():
        pp.set_pose(env.fg_object_id, env.PLACE_POSE)

        T_camera_to_world = mercury.geometry.look_at(
            env.PRE_PLACE_POSE[0], env.PLACE_POSE[0]
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
        # if pp.has_gui():
        #     imgviz.io.cv_imshow(
        #         np.hstack((rgb, imgviz.depth2rgb(depth))), "get_query_ocs"
        #     )
        #     imgviz.io.cv_waitkey(100)
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
        normals_in_world *= -1  # flip normals

        world_to_obj = pp.invert(pp.get_pose(env.fg_object_id))
        mask = segm == env.fg_object_id
        pcd_in_obj = mercury.geometry.transform_points(
            pcd_in_world[mask],
            mercury.geometry.transformation_matrix(*world_to_obj),
        )
        normals_in_obj = (
            mercury.geometry.transform_points(
                pcd_in_world[mask] + normals_in_world[mask],
                mercury.geometry.transformation_matrix(*world_to_obj),
            )
            - pcd_in_obj
        )

    return pcd_in_obj, normals_in_obj


def plan_and_execute_place(env, num_sample=5):
    pcd_in_obj, normals_in_obj = get_query_ocs(env)

    index = np.argmin(
        np.linalg.norm(pcd_in_obj - pcd_in_obj.mean(axis=0), axis=1)
    )
    point = pcd_in_obj[index]
    normal = normals_in_obj[index]

    obj_to_world = pp.get_pose(env.fg_object_id)
    point, point_p_normal = mercury.geometry.transform_points(
        [point, point + normal],
        mercury.geometry.transformation_matrix(*obj_to_world),
    )
    normal = point_p_normal - point

    obs = env.obs

    T_cam_to_world = mercury.geometry.look_at(
        eye=point + 0.1 * normal, target=point
    )
    fovy = np.deg2rad(80)
    height = 240
    width = 240
    rgb, depth, segm = mercury.pybullet.get_camera_image(
        T_cam_to_world,
        fovy,
        height,
        width,
    )
    # if pp.has_gui():
    #     imgviz.io.cv_imshow(
    #         np.hstack((rgb, imgviz.depth2rgb(depth))),
    #         "plan_and_execute_place",
    #     )
    #     imgviz.io.cv_waitkey(100)

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


def get_grasp_poses(env):
    segm = env.obs["segm"]
    depth = env.obs["depth"]

    K = env.ri.get_opengl_intrinsic_matrix()
    mask = segm == env.fg_object_id
    pcd_in_camera = mercury.geometry.pointcloud_from_depth(
        depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )
    normals_in_camera = mercury.geometry.normals_from_pointcloud(pcd_in_camera)
    pcd_in_camera = pcd_in_camera[mask]
    normals_in_camera = normals_in_camera[mask]

    camera_to_world = np.hsplit(env.obs["camera_to_world"], [3])
    pcd_in_world = mercury.geometry.transform_points(
        pcd_in_camera,
        mercury.geometry.transformation_matrix(*camera_to_world),
    )
    normals_in_world = (
        mercury.geometry.transform_points(
            pcd_in_camera + normals_in_camera,
            mercury.geometry.transformation_matrix(*camera_to_world),
        )
        - pcd_in_world
    )

    quaternion_in_world = mercury.geometry.quaternion_from_vec2vec(
        [0, 0, 1], normals_in_world
    )

    p = np.random.permutation(pcd_in_world.shape[0])

    obstacles = env.bg_objects + env.object_ids
    obstacles.remove(env.fg_object_id)
    for pose in zip(pcd_in_world[p], quaternion_in_world[p]):
        j = env.ri.solve_ik(pose, rotation_axis="z")
        if j is None:
            continue
        if not env.ri.validatej(
            j,
            obstacles=obstacles,
            min_distances=mercury.utils.StaticDict(-0.01),
        ):
            continue

        c = mercury.geometry.Coordinate(*pose)
        c.translate([0, 0, -0.1])
        j = env.ri.solve_ik(c.pose, n_init=1)
        if j is None:
            continue
        if not env.ri.validatej(
            j,
            obstacles=obstacles,
            min_distances=mercury.utils.StaticDict(-0.01),
        ):
            continue

        yield np.hstack(pose)


def plan_reorient(env, c_grasp, c_reorient):
    T_ee_af_to_world = c_grasp.matrix
    T_obj_af_to_world = c_reorient.matrix

    # lock_renderer = pp.LockRenderer()
    world_saver = pp.WorldSaver()

    result = {}

    result["grasp_pose"] = np.hstack(c_grasp.pose)
    result["reorient_pose"] = np.hstack(c_reorient.pose)

    obj_af = mercury.pybullet.duplicate(
        env.fg_object_id,
        texture=False,
        rgba_color=(0, 1, 0, 0.5),
        position=c_reorient.position,
        quaternion=c_reorient.quaternion,
    )

    def before_return():
        env.ri.attachments = []
        world_saver.restore()
        pp.remove_body(obj_af)
        # lock_renderer.restore()

    result["j_init"] = env.ri.getj()

    bg_object_ids = env.bg_objects + env.object_ids
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
        obstacles=env.bg_objects + env.object_ids,
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
    obstacles = env.bg_objects + env.object_ids
    obstacles.remove(env.ri.attachments[0].child)
    js = env.ri.planj(
        result["j_place"],
        obstacles=obstacles,
        min_distances={(env.ri.attachments[0].child, -1): -0.01},
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
    for _ in (_ for j in js for _ in env.ri.movej(j, timeout=1)):
        pp.step_simulation()
        time.sleep(pp.get_time_step())

    for _ in env.ri.grasp(
        min_dz=0.08, max_dz=0.12, rotation_axis=True, speed=0.001
    ):
        pp.step_simulation()
        time.sleep(pp.get_time_step())

    js = result["js_place"]
    for _ in (_ for j in js for _ in env.ri.movej(j, timeout=1, speed=0.005)):
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
            obstacles=env.bg_objects + env.object_ids,
            min_distances_start_goal=mercury.utils.StaticDict(-0.02),
        )
        if js is not None:
            for _ in (
                _ for j in js for _ in env.ri.movej(j, timeout=1, speed=0.005)
            ):
                pp.step_simulation()
                time.sleep(pp.get_time_step())

    for _ in env.ri.movej(env.ri.homej):
        pp.step_simulation()
        time.sleep(pp.get_time_step())


def get_static_reorient_poses(env):
    pcd_in_obj, normals_in_obj = get_query_ocs(env)
    index = np.argmin(
        np.linalg.norm(pcd_in_obj - pcd_in_obj.mean(axis=0), axis=1)
    )
    point_in_obj = pcd_in_obj[index]
    normal_in_obj = normals_in_obj[index]

    world_saver = pp.WorldSaver()
    lock_renderer = pp.LockRenderer()

    XY = [[0.2, -0.4]]
    pp.draw_aabb(([0.15, -0.45, 0.001], [0.25, -0.35, 0.001]))
    ABG = itertools.product(
        [0],
        [0],
        np.linspace(-np.pi, np.pi, num=16, endpoint=False),
    )

    # XY, ABG validation
    poses = []
    for (x, y), (a, b, g) in itertools.product(XY, ABG):
        c = mercury.geometry.Coordinate(position=(x, y, 0))
        c.quaternion = mercury.geometry.quaternion_from_euler((a, b, g))
        pp.set_pose(env.fg_object_id, c.pose)

        c.position[2] = -pp.get_aabb(env.fg_object_id)[0][2]
        pp.set_pose(env.fg_object_id, c.pose)

        points = pp.body_collision_info(
            env.fg_object_id, env.plane, max_distance=0.2
        )
        distance_to_plane = min(point[8] for point in points)
        assert distance_to_plane > 0
        c.position[2] -= distance_to_plane
        c.position[2] += 0.02
        pp.set_pose(env.fg_object_id, c.pose)

        if mercury.pybullet.is_colliding(env.fg_object_id):
            continue

        point, point_p_normal = mercury.geometry.transform_points(
            [point_in_obj, point_in_obj + normal_in_obj],
            mercury.geometry.transformation_matrix(*c.pose),
        )
        normal = point_p_normal - point
        angle = np.arccos(np.dot([0, 0, 1], normal))
        assert angle >= 0

        angle = np.arccos(
            np.dot([-1, 0], mercury.geometry.normalize_vec(normal[:2]))
        )
        assert angle >= 0
        if angle > np.deg2rad(45):
            continue

        poses.append(np.hstack(c.pose))
    poses = np.array(poses).reshape(-1, 7)

    world_saver.restore()
    lock_renderer.restore()

    return poses


def plan_place(env, target_grasp_poses):
    obj_to_world = pp.get_pose(env.fg_object_id)

    result = {}
    for grasp_pose in target_grasp_poses:
        with pp.LockRenderer(), pp.WorldSaver():
            ee_to_obj = np.hsplit(grasp_pose, [3])
            ee_to_world = pp.multiply(obj_to_world, ee_to_obj)
            j = env.ri.solve_ik(ee_to_world, rotation_axis="z")
            if j is None:
                logger.warning("j_grasp is not found")
                continue
            if not env.ri.validatej(
                j,
                obstacles=env.bg_objects,
                min_distances=mercury.utils.StaticDict(-0.01),
            ):
                logger.warning("j_grasp is invalid")
                continue
            result["j_grasp"] = j

            env.ri.setj(j)

            c = mercury.geometry.Coordinate(*ee_to_world)
            c.translate([0, 0, -0.1])
            j = env.ri.solve_ik(c.pose)
            if j is None:
                logger.warning("j_pre_grasp is not found")
                continue
            if not env.ri.validatej(
                j,
                obstacles=env.bg_objects,
                min_distances=mercury.utils.StaticDict(-0.01),
            ):
                logger.warning("j_pre_grasp is invalid")
                continue
            result["j_pre_grasp"] = j

            env.ri.setj(env.ri.homej)
            js = env.ri.planj(
                result["j_pre_grasp"],
                obstacles=env.bg_objects + env.object_ids,
                min_distances=mercury.utils.StaticDict(-0.01),
            )
            if js is None:
                logger.warning("js_pre_grasp is not found")
                continue
            result["js_pre_grasp"] = js

            env.ri.setj(result["j_grasp"])
            env.ri.attachments = [
                pp.Attachment(
                    env.ri.robot,
                    env.ri.ee,
                    pp.invert(ee_to_obj),
                    env.fg_object_id,
                )
            ]
            env.ri.attachments[0].assign()

            with env.ri.enabling_attachments():
                j = env.ri.solve_ik(
                    env.PRE_PLACE_POSE,
                    move_target=env.ri.robot_model.attachment_link0,
                )
                if j is None:
                    continue
                result["j_pre_place"] = j

                env.ri.setj(j)
                env.ri.attachments[0].assign()

                js = []
                for pose in pp.interpolate_poses(
                    env.PRE_PLACE_POSE, env.PLACE_POSE
                ):
                    j = env.ri.solve_ik(
                        pose,
                        move_target=env.ri.robot_model.attachment_link0,
                        n_init=1,
                    )
                    if j is None:
                        logger.warning("js_place is not found")
                        break
                    if not env.ri.validatej(j, obstacles=env.bg_objects):
                        j = None
                        logger.warning("js_place is invalid")
                        break
                    js.append(j)
                if j is None:
                    continue
                result["js_place"] = js
    return result


def execute_place(env, result):
    for _ in (_ for j in result["js_pre_grasp"] for _ in env.ri.movej(j)):
        pp.step_simulation()
        time.sleep(pp.get_time_step())

    for _ in env.ri.grasp(min_dz=0.08, max_dz=0.12, rotation_axis=True):
        pp.step_simulation()
        time.sleep(pp.get_time_step())

    for _ in env.ri.movej(env.ri.homej):
        pp.step_simulation()
        time.sleep(pp.get_time_step())

    for _ in env.ri.movej(result["j_pre_place"]):
        pp.step_simulation()
        time.sleep(pp.get_time_step())

    for _ in (
        _ for j in result["js_place"] for _ in env.ri.movej(j, speed=0.005)
    ):
        pp.step_simulation()
        time.sleep(pp.get_time_step())
