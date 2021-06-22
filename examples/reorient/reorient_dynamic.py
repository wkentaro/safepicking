#!/usr/bin/env python

import argparse
import itertools
import time

from loguru import logger
import numpy as np
import pybullet_planning as pp
import torch

import mercury

from _env import Env
import _reorient
import _utils

from pickable_eval import get_goal_oriented_reorient_poses


def plan_and_execute_reorient(
    env, grasp_poses, reorient_poses, visualize=True
):
    from legacy.train_reorientable import Model

    model = Model()
    model_file = "logs/reorientable/20210614_152113.921925-class_2_3_5_11_12_15-train_with_arbitrary_reorient_poses/models/model_best-epoch_0081.pth"  # NOQA
    logger.info(f"Loading {model_file}")
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    model.eval()
    model.cuda()

    obj_to_world = pp.get_pose(env.fg_object_id)
    world_to_obj = pp.invert(obj_to_world)
    grasp_poses = np.array(
        [
            np.hstack(pp.multiply(world_to_obj, np.hsplit(grasp_pose, [3])))
            for grasp_pose in grasp_poses
        ]
    )  # wrt world -> wrt obj

    N_grasp = grasp_poses.shape[0]
    N_reorient = reorient_poses.shape[0]
    B = N_grasp * N_reorient
    logger.info(f"N_grasp: {N_grasp}, N_reorient: {N_reorient}, B: {B}")

    grasp_poses = grasp_poses[:, None, :].repeat(N_reorient, axis=1)
    reorient_poses = reorient_poses[None, :, :].repeat(N_grasp, axis=0)

    grasp_poses = grasp_poses.reshape(B, -1).astype(np.float32)
    reorient_poses = reorient_poses.reshape(B, -1).astype(np.float32)

    object_fg_flags = []
    object_classes = []
    object_poses = []
    for object_id in env.object_ids:
        object_fg_flags.append(object_id == env.fg_object_id)
        object_classes.append(np.eye(22)[_utils.get_class_id(object_id)])
        object_poses.append(np.hstack(pp.get_pose(object_id)))
    object_fg_flags = np.stack(object_fg_flags, axis=0).astype(np.float32)
    object_classes = np.stack(object_classes, axis=0).astype(np.float32)
    object_poses = np.stack(object_poses, axis=0).astype(np.float32)

    object_fg_flags = np.tile(object_fg_flags[None], (B, 1))
    object_classes = np.tile(object_classes[None], (B, 1, 1))
    object_poses = np.tile(object_poses[None], (B, 1, 1))

    with torch.no_grad():
        solved_pred, length_pred = model(
            object_fg_flags=torch.as_tensor(object_fg_flags).cuda(),
            object_classes=torch.as_tensor(object_classes).cuda(),
            object_poses=torch.as_tensor(object_poses).cuda(),
            grasp_pose=torch.as_tensor(grasp_poses).cuda(),
            reorient_pose=torch.as_tensor(reorient_poses).cuda(),
        )
    solved_pred = solved_pred.cpu().numpy()
    solved_pred = solved_pred.sum(axis=1) / solved_pred.shape[1]

    indices = np.argsort(solved_pred)[::-1]

    result = {}
    for index in indices:
        ee_to_obj = grasp_poses[index, :3], grasp_poses[index, 3:]
        ee_to_world = pp.multiply(obj_to_world, ee_to_obj)
        obj_af_to_world = reorient_poses[index, :3], reorient_poses[index, 3:]

        if visualize:
            obj_af = mercury.pybullet.duplicate(
                env.fg_object_id,
                collision=False,
                rgba_color=(0, 1, 0, 0.5),
                position=obj_af_to_world[0],
                quaternion=obj_af_to_world[1],
            )
        else:
            lock_renderer = pp.LockRenderer()

        result = _reorient.plan_reorient(
            env,
            mercury.geometry.Coordinate(*ee_to_world),
            mercury.geometry.Coordinate(*obj_af_to_world),
        )

        if visualize:
            pp.remove_body(obj_af)
        else:
            lock_renderer.restore()

        if "js_place" in result:
            logger.success(f"solved_pred={solved_pred[index]:.1%}")
            break
        else:
            logger.warning(f"solved_pred={solved_pred[index]:.1%}")

    if "js_place" not in result:
        logger.error("No solution is found")
        return False

    _reorient.execute_plan(env, result)
    return True


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, help="seed", required=True)
    parser.add_argument(
        "--face",
        choices=["front", "right", "left"],
        default="front",
        help="face",
    )
    parser.add_argument("--mp4", help="mp4")
    args = parser.parse_args()

    env = Env(class_ids=[2, 3, 5, 11, 12, 15], mp4=args.mp4, face=args.face)
    env.random_state = np.random.RandomState(args.seed)
    env.eval = True
    env.reset()

    # if _reorient.plan_and_execute_place(env):
    #     return

    (
        reorient_poses,
        scores,
        target_grasp_poses,
    ) = get_goal_oriented_reorient_poses(env)

    reorient_poses = reorient_poses[scores > 0.7]
    reorient_poses = reorient_poses[
        np.random.permutation(reorient_poses.shape[0])
    ]

    grasp_poses = np.array(
        list(itertools.islice(_reorient.get_grasp_poses(env), 100))
    )

    plan_and_execute_reorient(env, grasp_poses, reorient_poses)

    for _ in range(480):
        pp.step_simulation()
        time.sleep(pp.get_time_step())

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

    if "js_place" not in result:
        logger.error("Failed to plan pick-and-place")
    else:
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


if __name__ == "__main__":
    main()
