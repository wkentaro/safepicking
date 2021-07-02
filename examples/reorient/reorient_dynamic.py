#!/usr/bin/env python

import argparse
import itertools
import json
import time

from loguru import logger
import numpy as np
import path
import pybullet_planning as pp
import torch

import mercury

from _env import Env
import _reorient
import _utils

from pickable_eval import get_goal_oriented_reorient_poses
from reorientable_train import Model


def plan_dynamic_reorient(
    env, grasp_poses, reorient_poses, pickable, visualize=True
):
    model = Model()
    model_file = "logs/reorientable/20210630_024141.842308-train_size_4500/models/model_best-epoch_0077.pt"  # NOQA
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

    for threshold in np.linspace(0.9, 0.1, num=10):
        keep = reorientable_pred[:, 2] > threshold
        if keep.sum() > 10:
            pickable = pickable[keep]
            reorientable_pred = reorientable_pred[keep]
            trajectory_length_pred = trajectory_length_pred[keep]
            grasp_poses = grasp_poses[keep]
            reorient_poses = reorient_poses[keep]
            break

    indices1 = np.argsort(trajectory_length_pred)
    indices2 = np.argsort(reorientable_pred[:, 2])[::-1]
    indices = np.r_[indices1[:3], indices2[:3]]

    assert (
        pickable.shape[0]
        == reorientable_pred.shape[0]
        == trajectory_length_pred.shape[0]
        == grasp_poses.shape[0]
        == reorient_poses.shape[0]
    )

    result = {}
    for index in indices:
        ee_to_obj = np.hsplit(grasp_poses[index], [3])
        ee_to_world = pp.multiply(obj_to_world, ee_to_obj)
        obj_af_to_world = np.hsplit(reorient_poses[index], [3])

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
                f"pickable={pickable[index]:.1%}, "
                f"graspable_pred={reorientable_pred[index, 0]:.1%}, "
                f"placable_pred={reorientable_pred[index, 1]:.1%}, "
                f"reorientable_pred={reorientable_pred[index, 2]:.1%}, "
                f"trajectory_length_pred={trajectory_length_pred[index]:.1f}, "
                f"trajectory_length_true={result['js_place_length']:.1f}"
            )
            break
        else:
            logger.warning(
                f"pickable={pickable[index]:.1%}, "
                f"graspable_pred={reorientable_pred[index, 0]:.1%}, "
                f"placable_pred={reorientable_pred[index, 1]:.1%}, "
                f"reorientable_pred={reorientable_pred[index, 2]:.1%}, "
                f"trajectory_length_pred={trajectory_length_pred[index]:.1f}"
            )

    return result


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
    parser.add_argument("--nogui", action="store_true", help="no gui")
    args = parser.parse_args()

    json_file = path.Path(
        f"logs/reorient_dynamic/{args.seed:08d}-{args.face}.json"
    )
    if args.nogui and json_file.exists():
        logger.info(f"Already json_file exists: {json_file}")
        return

    env = Env(
        class_ids=[2, 3, 5, 11, 12, 15],
        gui=not args.nogui,
        mp4=args.mp4,
        face=args.face,
    )
    env.random_state = np.random.RandomState(args.seed)
    env.eval = True
    env.reset()

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
        for threshold in np.linspace(0.9, 0.1, num=10):
            indices = np.where(pickable > threshold)[0]
            if indices.size > 1000:
                break
        indices = np.random.choice(indices, 1000, replace=False)
        reorient_poses = reorient_poses[indices]
        pickable = pickable[indices]

        result = plan_dynamic_reorient(
            env, grasp_poses, reorient_poses, pickable
        )

    if "js_place" not in result:
        logger.error("No solution is found")
        success_reorient = False
        execution_time = np.nan
        trajectory_length = np.nan
        success = False
    else:
        exec_result = _reorient.execute_reorient(env, result)
        success_reorient = True
        execution_time = exec_result["t_place"]
        trajectory_length = result["js_place_length"]

        for _ in range(480):
            pp.step_simulation()
            if pp.has_gui():
                time.sleep(pp.get_time_step())

        result = _reorient.plan_place(env, target_grasp_poses)
        if "js_place" not in result:
            logger.error("Failed to plan pick-and-place")
            success = False
        else:
            _reorient.execute_place(env, result)
            success = True

    if args.nogui:
        json_file.parent.makedirs_p()
        with open(json_file, "w") as f:
            json.dump(
                dict(
                    success_reorient=success_reorient,
                    success=success,
                    trajectory_length=trajectory_length,
                    execution_time=execution_time,
                ),
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
