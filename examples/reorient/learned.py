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
from planned import plan_and_execute_place
from planned import plan_reorient
from reorient_poses import get_reorient_poses2
from train import Model


def plan_and_execute_reorient(env, model, timeout, visualize=True):
    obj_to_world = pp.get_pose(env.fg_object_id)
    world_to_obj = pp.invert(obj_to_world)
    grasp_poses = np.array(list(itertools.islice(get_grasp_poses(env), 32)))
    grasp_poses = np.array(
        [
            np.hstack(pp.multiply(world_to_obj, (p[:3], p[3:])))
            for p in grasp_poses
        ]
    )  # wrt world -> wrt obj

    reorient_poses, angles, _, _ = get_reorient_poses2(env)

    p = env.random_state.permutation(reorient_poses.shape[0])[:10000]
    reorient_poses = reorient_poses[p]
    angles = angles[p]

    N_grasp = grasp_poses.shape[0]
    N_reorient = reorient_poses.shape[0]
    B = N_grasp * N_reorient
    logger.info(f"N_grasp: {N_grasp}, N_reorient: {N_reorient}, B: {B}")

    grasp_poses = grasp_poses[:, None, :].repeat(N_reorient, axis=1)
    reorient_poses = reorient_poses[None, :, :].repeat(N_grasp, axis=0)
    angles = angles[None, :].repeat(N_grasp, axis=0)

    grasp_poses = grasp_poses.reshape(B, -1).astype(np.float32)
    reorient_poses = reorient_poses.reshape(B, -1).astype(np.float32)
    angles = angles.reshape(B).astype(np.float32)

    object_fg_flags = []
    object_classes = []
    object_poses = []
    for object_id in env.object_ids:
        object_fg_flags.append(object_id == env.fg_object_id)
        object_classes.append(np.eye(22)[common_utils.get_class_id(object_id)])
        object_poses.append(np.hstack(pp.get_pose(object_id)))
    object_fg_flags = np.stack(object_fg_flags, axis=0).astype(np.float32)
    object_classes = np.stack(object_classes, axis=0).astype(np.float32)
    object_poses = np.stack(object_poses, axis=0).astype(np.float32)

    object_fg_flags = np.tile(object_fg_flags[None], (B, 1))
    object_classes = np.tile(object_classes[None], (B, 1, 1))
    object_poses = np.tile(object_poses[None], (B, 1, 1))

    with torch.no_grad():
        solved_pred = model(
            object_fg_flags=torch.as_tensor(object_fg_flags).cuda(),
            object_classes=torch.as_tensor(object_classes).cuda(),
            object_poses=torch.as_tensor(object_poses).cuda(),
            grasp_pose=torch.as_tensor(grasp_poses).cuda(),
            reorient_pose=torch.as_tensor(reorient_poses).cuda(),
        )
    solved_pred = solved_pred.cpu().numpy()
    solved_pred = solved_pred.sum(axis=1) / solved_pred.shape[1]

    keep = solved_pred >= 0.90

    grasp_poses = grasp_poses[keep]
    reorient_poses = reorient_poses[keep]
    angles = angles[keep]
    solved_pred = solved_pred[keep]

    indices = np.argsort(angles)

    t_start = time.time()

    results = []
    for index in indices:
        logger.info(
            f"solved_pred={solved_pred[index]:.1%}, "
            f"angle={np.rad2deg(angles[index]):.1f} [deg]"
        )
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

        result = plan_reorient(
            env,
            mercury.geometry.Coordinate(*ee_to_world),
            mercury.geometry.Coordinate(*obj_af_to_world),
        )

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
    parser.add_argument("--timeout", type=int, default=3, help="timeout")
    parser.add_argument("--visualize", action="store_true", help="visualize")
    parser.add_argument("--mp4", help="mp4")
    args = parser.parse_args()

    env = PickAndPlaceEnv(class_ids=args.class_ids, mp4=args.mp4)
    env.eval = True
    env.random_state = np.random.RandomState(args.seed)
    env.launch()
    with pp.LockRenderer():
        env.reset()

    common_utils.pause(args.pause)

    model = Model()
    model_file = sorted(args.log_dir.glob("models/model_best-epoch_*.pth"))[-1]
    logger.info(f"Loading {model_file}")
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    model.eval()
    model.cuda()

    while True:
        if plan_and_execute_place(env):
            return

        if not plan_and_execute_reorient(
            env,
            model=model,
            timeout=args.timeout,
            visualize=args.visualize,
        ):
            break

        # for next plan_and_execute_reorient
        env.setj_to_camera_pose()
        env.update_obs()


if __name__ == "__main__":
    main()
