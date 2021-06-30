#!/usr/bin/env python

import argparse

import imgviz
import numpy as np
import pybullet_planning as pp

import mercury

from _agent import DqnAgent
from _env import PickFromPileEnv


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=[
            "closedloop_pose_net",
            "openloop_pose_net",
            "conv_net",
            "position_conv_net",
            "pose_conv_net",
            "position_pose_conv_net",
            "fusion_net",
        ],
        help="model",
    )
    parser.add_argument("--nogui", action="store_true", help="no gui")
    parser.add_argument("--draw-obs", action="store_true", help="draw obs")
    args = parser.parse_args()

    env = PickFromPileEnv(gui=not args.nogui, action_frame="ee")
    obs = env.reset()

    if args.draw_obs:
        viz = imgviz.tile(
            [
                imgviz.depth2rgb(obs["heightmap"]),
                np.uint8(obs["maskmap"]) * 255,
                np.uint8(imgviz.normalize(obs["positionmap"]) * 255),
                np.uint8(imgviz.normalize(obs["posemap"]) * 255),
            ],
            shape=(1, 4),
        )
        imgviz.io.pyglet_imshow(viz)
        imgviz.io.pyglet_run()

    if args.model is None:
        agent = None
    else:
        agent = DqnAgent(env=env, model=args.model)
        agent.build(training=False)
        print(sum(p.nelement() for p in agent.q.parameters()))

    while True:
        act_result = agent.act(
            step=-1,
            observation={k: v[None, None] for k, v in obs.items()},
            deterministic=False,
            env=env,
        )
        transition = env.step(act_result)
        if transition.terminal:
            if args.draw_obs:
                pp.draw_pose(np.hsplit(env.ee_pose_init, [3]))
                grasp_flags = obs["grasp_flags_init"]
                object_labels = obs["object_labels_init"]
                object_poses = obs["object_poses_init"]

                object_poses[:, :3] += [
                    env.ee_pose_init[0],
                    env.ee_pose_init[1],
                    0,
                ]

                for grasp_flag, object_label, object_pose in zip(
                    grasp_flags, object_labels, object_poses
                ):
                    class_id = env.CLASS_IDS[np.argwhere(object_label)[0, 0]]
                    visual_file = mercury.datasets.ycb.get_visual_file(
                        class_id=class_id
                    )
                    mercury.pybullet.create_mesh_body(
                        visual_file=visual_file,
                        texture=False,
                        rgba_color=(1, 0, 0, 0.5)
                        if grasp_flag
                        else (0, 0, 1, 0.5),
                        position=object_pose[:3],
                        quaternion=object_pose[3:],
                        mesh_scale=(1.05, 1.05, 1.05),
                    )

                ee_poses = obs["ee_poses"]
                ee_poses[:, :3] += [
                    env.ee_pose_init[0],
                    env.ee_pose_init[1],
                    0,
                ]
                for pose in ee_poses:
                    if (pose == 0).all():
                        continue
                    pp.draw_pose(np.hsplit(pose, [3]))
                while True:
                    pp.step_simulation()

            obs = env.reset()
        else:
            obs = transition.observation


if __name__ == "__main__":
    main()
