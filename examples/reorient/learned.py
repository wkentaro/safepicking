#!/usr/bin/env python

import argparse
import itertools
import time

from loguru import logger
import numpy as np
import path
import pybullet_planning as pp
import torch

import mercury

from pick_and_place_env import PickAndPlaceEnv

import common_utils
from planned import execute_plan
from planned import get_grasp_poses
from planned import get_reorient_poses
from planned import plan_and_execute_place
from planned import plan_reorient
from train import Dataset
from train import Model


def plan_and_execute_reorient(env, model, nolearning, timeout, visualize=True):
    grasp_pose = []
    initial_pose = []
    reorient_pose = []
    c_initial = mercury.geometry.Coordinate(*pp.get_pose(env.fg_object_id))
    c_reorients = list(get_reorient_poses(env, discretize=False))
    c_grasps = list(itertools.islice(get_grasp_poses(env), 32))
    for c_reorient, c_grasp in itertools.product(c_reorients, c_grasps):
        grasp_pose.append(np.hstack(c_grasp.pose))
        initial_pose.append(np.hstack(c_initial.pose))
        reorient_pose.append(np.hstack(c_reorient.pose))
    grasp_pose = np.stack(grasp_pose, axis=0).astype(np.float32)
    initial_pose = np.stack(initial_pose, axis=0).astype(np.float32)
    reorient_pose = np.stack(reorient_pose, axis=0).astype(np.float32)

    B = reorient_pose.shape[0]

    object_classes = []
    object_poses = []
    for object_id in env.object_ids:
        class_id = common_utils.get_class_id(object_id)
        object_classes.append(np.eye(22)[class_id])
        object_poses.append(np.hstack(pp.get_pose(object_id)))
    object_classes = np.stack(object_classes, axis=0).astype(np.float32)
    object_poses = np.stack(object_poses, axis=0).astype(np.float32)

    object_classes = np.tile(object_classes[None], (B, 1, 1))
    object_poses = np.tile(object_poses[None], (B, 1, 1))

    object_class = common_utils.get_class_id(env.fg_object_id)
    object_class = np.eye(22)[object_class].astype(np.float32)
    object_class = np.tile(object_class[None], (B, 1))

    if nolearning:
        success_pred = np.full(B, np.nan)
        length_pred = np.full(B, np.nan)
        auc_pred = np.full(B, np.nan)
        indices = np.arange(B)
    else:
        with torch.no_grad():
            success_pred, length_pred, auc_pred = model(
                object_classes=torch.as_tensor(object_classes),
                object_poses=torch.as_tensor(object_poses),
                object_class=torch.as_tensor(object_class),
                grasp_pose=torch.as_tensor(grasp_pose),
                initial_pose=torch.as_tensor(initial_pose),
                reorient_pose=torch.as_tensor(reorient_pose),
            )
        success_pred = success_pred.cpu().numpy()
        length_pred = (
            length_pred.cpu().numpy() * Dataset.JS_PLACE_LENGTH_SCALING
        )
        auc_pred = auc_pred.cpu().numpy()

        for threshold in [0.6, 0.4, 0.2, 0]:
            keep = success_pred > threshold
            if keep.sum() > 0:
                break

        success_pred = success_pred[keep]
        length_pred = length_pred[keep]
        auc_pred = auc_pred[keep]

        grasp_pose = grasp_pose[keep]
        reorient_pose = reorient_pose[keep]

        length_pred_normalized = (
            length_pred - length_pred.mean()
        ) / length_pred.std()
        auc_pred_normalized = (auc_pred - auc_pred.mean()) / auc_pred.std()

        metric = length_pred_normalized - auc_pred_normalized
        indices = np.argsort(metric)

    t_start = time.time()

    results = []
    for index in indices:
        logger.info(
            f"success_pred={success_pred[index]:.1%}, "
            f"length_pred={length_pred[index]:.2f}, "
            f"auc_pred={auc_pred[index]:.2f}, "
        )
        c_grasp = mercury.geometry.Coordinate(
            grasp_pose[index, :3], grasp_pose[index, 3:]
        )
        c_reorient = mercury.geometry.Coordinate(
            reorient_pose[index, :3], reorient_pose[index, 3:]
        )

        if visualize:
            obj_af = mercury.pybullet.duplicate(
                env.fg_object_id,
                collision=False,
                rgba_color=(0, 1, 0, 0.5),
                position=c_reorient.position,
                quaternion=c_reorient.quaternion,
            )
        else:
            lock_renderer = pp.LockRenderer()

        result = plan_reorient(env, c_grasp, c_reorient)

        if visualize:
            pp.remove_body(obj_af)
        else:
            lock_renderer.restore()

        if "js_place_length" in result:
            results.append(result)

        if (time.time() - t_start) > timeout:
            break
    if not results:
        logger.error("No solution is found")
        return False

    result = min(results, key=lambda x: x["js_place_length"])
    logger.info(f"length_true={result['js_place_length']:.2f}")

    execute_plan(env, result)
    return True


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("log_dir", type=path.Path, help="log dir")
    parser.add_argument("--class-ids", type=int, nargs="+", help="class ids")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--pause", action="store_true", help="pause")
    parser.add_argument("--nolearning", action="store_true", help="nolearning")
    parser.add_argument("--timeout", type=int, default=3, help="timeout")
    parser.add_argument("--visualize", action="store_true", help="visualize")
    parser.add_argument("--mp4", help="mp4")
    args = parser.parse_args()

    env = PickAndPlaceEnv(class_ids=args.class_ids, mp4=args.mp4)
    env.eval = True
    env.random_state = np.random.RandomState(args.seed)
    env.reset()

    common_utils.pause(args.pause)

    model = Model()
    model_file = sorted(args.log_dir.glob("models/model_best-epoch_*.pth"))[-1]
    logger.info(f"Loading {model_file}")
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    model.eval()

    while True:
        if plan_and_execute_place(env):
            return

        if not plan_and_execute_reorient(
            env,
            model=model,
            nolearning=args.nolearning,
            timeout=args.timeout,
            visualize=args.visualize,
        ):
            break


if __name__ == "__main__":
    main()
