#!/usr/bin/env python

import argparse

import numpy as np
import path
import pybullet_planning as pp
import torch

import mercury

from _env import Env
import _reorient
import _utils

from pickable_reorient_poses import get_reorient_poses
from pickable_train import Model


home = path.Path("~").expanduser()


def get_goal_oriented_reorient_poses(env):
    model = Model()
    model.load_state_dict(
        torch.load(
            "./logs/pickable/20210619_025859.533125-train_size_1000/models/model_best-epoch_0191.pth"  # NOQA
        )
    )
    model.cuda()

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
    object_fg_flags = np.array(object_fg_flags, dtype=bool)
    object_labels = np.array(object_labels, dtype=np.float32)
    object_poses = np.array(object_poses, dtype=np.float32)

    reorient_poses = get_reorient_poses(env)

    # target grasp_pose
    pcd_in_obj, normals_in_obj = _reorient.get_query_ocs(env)
    indices = np.random.permutation(pcd_in_obj.shape[0])[:10]

    pcd_in_obj = pcd_in_obj[indices]
    normals_in_obj = normals_in_obj[indices]
    quaternion_in_obj = mercury.geometry.quaternion_from_vec2vec(
        [0, 0, -1], normals_in_obj
    )
    grasp_pose = np.hstack([pcd_in_obj, quaternion_in_obj])

    for pose in grasp_pose:
        pp.draw_pose(
            np.hsplit(pose, [3]),
            parent=env.fg_object_id,
            length=0.05,
            width=3,
        )

    R = reorient_poses.shape[0]
    G = grasp_pose.shape[0]
    O = object_fg_flags.shape[0]

    object_fg_flags = object_fg_flags[None].repeat(R, axis=0)
    object_labels = object_labels[None].repeat(R, axis=0)
    object_poses = object_poses[None].repeat(R, axis=0)

    object_poses[object_fg_flags] = reorient_poses

    object_fg_flags = object_fg_flags[:, None].repeat(G, axis=1)
    object_labels = object_labels[:, None].repeat(G, axis=1)
    object_poses = object_poses[:, None].repeat(G, axis=1)
    grasp_pose = grasp_pose[None].repeat(R, axis=0)

    object_fg_flags = object_fg_flags.reshape(R * G, O)
    object_labels = object_labels.reshape(R * G, O, 7)
    object_poses = object_poses.reshape(R * G, O, 7)
    grasp_pose = grasp_pose.reshape(R * G, 7)

    with torch.no_grad():
        pickable_pred = model(
            object_fg_flags=torch.as_tensor(object_fg_flags).float().cuda(),
            object_labels=torch.as_tensor(object_labels).float().cuda(),
            object_poses=torch.as_tensor(object_poses).float().cuda(),
            grasp_pose=torch.as_tensor(grasp_pose).float().cuda(),
        )
    pickable_pred = pickable_pred.cpu().numpy()

    pickable_pred = pickable_pred.reshape(R, G).mean(axis=1)
    grasp_pose = grasp_pose.reshape(R, G, 7)

    return reorient_poses, pickable_pred, grasp_pose[0]


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, required=True, help="seed")
    parser.add_argument(
        "--face",
        choices=["front", "right", "left"],
        default="front",
        help="face",
    )
    parser.add_argument("--visualize", action="store_true", help="visualize")
    args = parser.parse_args()

    env = Env(class_ids=[2, 3, 5, 11, 12, 15], gui=True, face=args.face)
    env.random_state = np.random.RandomState(args.seed)
    env.eval = True
    env.reset()

    reorient_poses, scores, grasp_poses = get_goal_oriented_reorient_poses(env)

    reorient_poses = reorient_poses[np.argsort(scores)[::-1]]

    if args.visualize:
        for reorient_pose in reorient_poses:
            mercury.pybullet.duplicate(
                env.fg_object_id,
                position=reorient_pose[:3],
                quaternion=reorient_pose[3:],
                collision=False,
            )
        return

    for reorient_pose in reorient_poses:
        mercury.pybullet.pause()

        env.ri.setj(env.ri.homej)

        obj_af_to_world = np.hsplit(reorient_pose, [3])
        pp.set_pose(env.fg_object_id, obj_af_to_world)

        for _ in range(480):
            pp.step_simulation()

        for ee_af_to_obj in grasp_poses:
            ee_af_to_obj = np.hsplit(ee_af_to_obj, [3])
            ee_af_to_world = pp.multiply(
                pp.get_pose(env.fg_object_id), ee_af_to_obj
            )
            j = env.ri.solve_ik(
                ee_af_to_world, rotation_axis="z", rthre=np.deg2rad(10)
            )
            if j is not None:
                obstacles = [env.plane] + env.object_ids
                obstacles.remove(env.fg_object_id)
                if not env.ri.validatej(j, obstacles=obstacles):
                    j = None
            if j is not None:
                env.ri.setj(j)
                break
        else:
            print("Failed to solve IK")


if __name__ == "__main__":
    main()
