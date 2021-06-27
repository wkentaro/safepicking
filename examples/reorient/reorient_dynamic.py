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
from reorientable_train import Model


def plan_and_execute_reorient(
    env, grasp_poses, reorient_poses, pickable, visualize=True
):
    model = Model()
    model_file = "logs/reorientable/20210627_191451.547701-smooth_l1_loss-lambda1_1-lambda2_1/models/model_best-epoch_0021.pt"  # NOQA
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

    # pose representation -> point-normal representation
    grasp_points = []
    for grasp_pose in grasp_poses:
        ee_to_obj = np.hsplit(grasp_pose, [3])
        grasp_point_start = ee_to_obj[0]
        grasp_point_end = mercury.geometry.transform_points(
            [[0, 0, 1]], mercury.geometry.transformation_matrix(*ee_to_obj)
        )[0]
        grasp_points.append(np.hstack([grasp_point_start, grasp_point_end]))

        # pp.draw_pose(
        #     np.hsplit(grasp_pose, [3]),
        #     parent=env.fg_object_id,
        #     length=0.05,
        #     width=3,
        # )
    grasp_points = np.array(grasp_points)

    N_grasp = grasp_points.shape[0]
    N_reorient = reorient_poses.shape[0]
    B = N_grasp * N_reorient
    logger.info(f"N_grasp: {N_grasp}, N_reorient: {N_reorient}, B: {B}")

    grasp_poses = grasp_poses[:, None, :].repeat(N_reorient, axis=1)
    grasp_points = grasp_points[:, None, :].repeat(N_reorient, axis=1)
    reorient_poses = reorient_poses[None, :, :].repeat(N_grasp, axis=0)
    pickable = pickable[None, :].repeat(N_grasp, axis=0)

    grasp_poses = grasp_poses.reshape(B, -1).astype(np.float32)
    grasp_points = grasp_points.reshape(B, -1).astype(np.float32)
    reorient_poses = reorient_poses.reshape(B, -1).astype(np.float32)
    pickable = pickable.reshape(B).astype(np.float32)

    class_ids = [2, 3, 5, 11, 12, 15, 16]

    object_fg_flags = []
    object_labels = []
    object_poses = []
    for object_id in env.object_ids:
        object_fg_flags.append(object_id == env.fg_object_id)
        object_label = np.zeros(7)
        object_label[class_ids.index(_utils.get_class_id(object_id))] = 1
        object_labels.append(object_label)
        object_poses.append(np.hstack(pp.get_pose(object_id)))
    object_fg_flags = np.stack(object_fg_flags, axis=0).astype(np.float32)
    object_labels = np.stack(object_labels, axis=0).astype(np.float32)
    object_poses = np.stack(object_poses, axis=0).astype(np.float32)

    object_fg_flags = np.tile(object_fg_flags[None], (B, 1))
    object_labels = np.tile(object_labels[None], (B, 1, 1))
    object_poses = np.tile(object_poses[None], (B, 1, 1))

    with torch.no_grad():
        reorientable_pred, trajectory_length_pred = model(
            object_fg_flags=torch.as_tensor(object_fg_flags).cuda(),
            object_labels=torch.as_tensor(object_labels).cuda(),
            object_poses=torch.as_tensor(object_poses).cuda(),
            grasp_pose=torch.as_tensor(grasp_points).cuda(),
            reorient_pose=torch.as_tensor(reorient_poses).cuda(),
        )
    reorientable_pred = reorientable_pred.cpu().numpy()
    trajectory_length_pred = trajectory_length_pred.cpu().numpy()

    keep = (reorientable_pred[:, 0] > 0.75) & (reorientable_pred[:, 1] > 0.75)
    grasp_poses = grasp_poses[keep]
    reorientable_pred = reorientable_pred[keep]
    trajectory_length_pred = trajectory_length_pred[keep]

    reorientable_pred = reorientable_pred[:, 2]
    indices = np.argsort(reorientable_pred)[-100:]

    grasp_poses = grasp_poses[indices]
    reorientable_pred = reorientable_pred[indices]
    trajectory_length_pred = trajectory_length_pred[indices]

    indices = np.argsort(trajectory_length_pred)

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
            env, np.hstack(ee_to_world), np.hstack(obj_af_to_world)
        )

        if visualize:
            pp.remove_body(obj_af)
        else:
            lock_renderer.restore()

        if "js_place" in result:
            logger.success(
                f"reorientable_pred={reorientable_pred[index]:.1%}, "
                f"pickable={pickable[index]:.1%}"
            )
            break
        else:
            logger.warning(
                f"reorientable_pred={reorientable_pred[index]:.1%}, "
                f"pickable={pickable[index]:.1%}"
            )

    if "js_place" not in result:
        logger.error("No solution is found")
        return False

    _reorient.execute_reorient(env, result)
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

    if 0:
        pcd_in_obj, normals_in_obj = _reorient.get_query_ocs(env)
        quaternion_in_obj = mercury.geometry.quaternion_from_vec2vec(
            [0, 0, -1], normals_in_obj
        )
        target_grasp_poses = np.hstack([pcd_in_obj, quaternion_in_obj])
        target_grasp_poses = target_grasp_poses[
            np.random.permutation(target_grasp_poses.shape[0])
        ][:10]

        result = _reorient.plan_place(env, target_grasp_poses)
        if "js_place" not in result:
            logger.error("Failed to plan pick-and-place")
        else:
            _reorient.execute_place(env, result)

    (
        reorient_poses,
        pickable,
        target_grasp_poses,
    ) = get_goal_oriented_reorient_poses(env)

    grasp_poses = np.array(
        list(itertools.islice(_reorient.get_grasp_poses(env), 100))
    )

    if 0:
        for reorient_pose in reorient_poses[np.argsort(pickable)[::-1]]:
            grasp_pose = grasp_poses[
                np.random.permutation(grasp_poses.shape[0])[0]
            ]
            result = _reorient.plan_reorient(env, grasp_pose, reorient_pose)
            if "js_place" in result:
                break
        _reorient.execute_reorient(env, result)
    else:
        indices = np.argsort(pickable)[::-1][:1000]
        reorient_poses = reorient_poses[indices]
        pickable = pickable[indices]

        plan_and_execute_reorient(env, grasp_poses, reorient_poses, pickable)

    for _ in range(480):
        pp.step_simulation()
        time.sleep(pp.get_time_step())

    result = _reorient.plan_place(env, target_grasp_poses)
    if "js_place" not in result:
        logger.error("Failed to plan pick-and-place")
    else:
        _reorient.execute_place(env, result)


if __name__ == "__main__":
    main()
