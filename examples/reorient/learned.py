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


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("log_dir", type=path.Path, help="log dir")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--pause", action="store_true", help="pause")
    parser.add_argument("--nolearning", action="store_true", help="nolearning")
    parser.add_argument(
        "--num-sample", type=int, default=10, help="num sample"
    )
    args = parser.parse_args()

    env = PickAndPlaceEnv()
    env.eval = True
    env.random_state = np.random.RandomState(args.seed)
    env.reset()

    common_utils.pause(args.pause)

    model = Model()
    model_file = sorted(args.log_dir.glob("models/model_best-epoch_*.pth"))[-1]
    logger.info(f"Loading {model_file}")
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    model.eval()

    t_start = time.time()

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

    if args.nolearning:
        success_pred = np.full(B, np.nan)
        length_pred = np.full(B, np.nan)
        auc_pred = np.full(B, np.nan)
        indices = np.arange(B)
    else:
        with torch.no_grad():
            success_pred, length_pred, auc_pred = model(
                object_classes=torch.as_tensor(object_classes),
                object_poses=torch.as_tensor(object_poses),
                grasp_pose=torch.as_tensor(grasp_pose),
                initial_pose=torch.as_tensor(initial_pose),
                reorient_pose=torch.as_tensor(reorient_pose),
            )
        success_pred = success_pred.cpu().numpy()
        length_pred = (
            length_pred.cpu().numpy() * Dataset.JS_PLACE_LENGTH_SCALING
        )
        auc_pred = auc_pred.cpu().numpy()

        keep = success_pred > 0.6

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

    results = []
    for index in indices[: args.num_sample]:
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

        obj_af = mercury.pybullet.duplicate(
            env.fg_object_id,
            collision=False,
            texture=False,
            rgba_color=(0, 1, 0, 0.5),
            position=c_reorient.position,
            quaternion=c_reorient.quaternion,
        )

        result = plan_reorient(env, c_grasp, c_reorient)

        pp.remove_body(obj_af)

        if "js_place_length" in result:
            results.append(result)
            if args.num_sample < 0:
                if len(results) == abs(args.num_sample):
                    break
    if not results:
        logger.error("No solution is found")
        return

    result = min(results, key=lambda x: x["js_place_length"])
    logger.info(f"length_true={result['js_place_length']:.2f}")

    logger.info(f"planning_time={time.time() - t_start:.2f} [s]")

    execute_plan(env, result)

    plan_and_execute_place(env)


if __name__ == "__main__":
    main()
