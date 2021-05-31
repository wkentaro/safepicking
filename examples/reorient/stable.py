#!/usr/bin/env python

import argparse
import time

from loguru import logger
import numpy as np
import path
import pybullet_planning as pp
import torch

import mercury

from pick_and_place_env import PickAndPlaceEnv

import common_utils
from planned import get_reorient_poses
from train import Model


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("log_dir", type=path.Path, help="log dir")
    parser.add_argument(
        "--class-ids", type=int, nargs="+", help="class ids", required=True
    )
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--pause", action="store_true", help="pause")
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

    # -------------------------------------------------------------------------

    reorient_pose = []
    c_reorients = list(get_reorient_poses(env, discretize=False))
    for c_reorient in c_reorients:
        reorient_pose.append(np.hstack(c_reorient.pose))
    reorient_pose = np.stack(reorient_pose, axis=0).astype(np.float32)

    B = reorient_pose.shape[0]

    object_fg_flags = []
    object_classes = []
    object_poses = []
    for object_id in env.object_ids:
        class_id = common_utils.get_class_id(object_id)
        object_fg_flags.append(object_id == env.fg_object_id)
        object_classes.append(np.eye(22)[class_id])
        object_poses.append(np.hstack(pp.get_pose(object_id)))
    object_fg_flags = np.stack(object_fg_flags, axis=0)
    object_classes = np.stack(object_classes, axis=0).astype(np.float32)
    object_poses = np.stack(object_poses, axis=0).astype(np.float32)

    object_fg_flags = np.tile(object_fg_flags[None], (B, 1))
    object_classes = np.tile(object_classes[None], (B, 1, 1))
    object_poses = np.tile(object_poses[None], (B, 1, 1))

    object_poses[object_fg_flags] = reorient_pose
    object_fg_flags = object_fg_flags.astype(np.float32)

    with torch.no_grad():
        stable_pred = model(
            object_fg_flags=torch.as_tensor(object_fg_flags),
            object_classes=torch.as_tensor(object_classes),
            object_poses=torch.as_tensor(object_poses),
        )
    stable_pred = stable_pred.cpu().numpy()

    indices = np.argsort(stable_pred)[::-1]

    mercury.pybullet.duplicate(
        env.fg_object_id,
        collision=False,
        position=pp.get_pose(env.fg_object_id)[0],
        quaternion=pp.get_pose(env.fg_object_id)[1],
        rgba_color=(0, 1, 0, 0.5),
    )

    if 0:
        while True:
            for index in indices:
                if stable_pred[index] > 0:
                    print(stable_pred[index])
                    pose = reorient_pose[index][:3], reorient_pose[index][3:]
                    pp.set_pose(env.fg_object_id, pose)
                    text = pp.add_text(
                        f"{stable_pred[index]:.3f}",
                        position=pose[0] + [0, 0, 0.1],
                    )
                    time.sleep(0.1)
                    pp.remove_debug(text)

    for index in indices:
        print(stable_pred[index])

        with pp.WorldSaver():
            pose = reorient_pose[index][:3], reorient_pose[index][3:]
            pp.set_pose(env.fg_object_id, pose)

            text = pp.add_text(
                f"{stable_pred[index]:.3f}", position=pose[0] + [0, 0, 0.1]
            )

            time.sleep(1)

            for _ in range(int(round(3 / pp.get_time_step()))):
                pp.step_simulation()
                time.sleep(pp.get_time_step())

            pp.remove_debug(text)


if __name__ == "__main__":
    main()
