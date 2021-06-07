#!/usr/bin/env python

import argparse
import pprint

import numpy as np
import pybullet_planning as pp

import mercury

from agent import DqnAgent
from env import PickFromPileEnv


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    choices = ["closedloop_pose_net", "openloop_pose_net"]
    parser.add_argument(
        "--model",
        default=choices[0],
        choices=choices,
        help="model",
    )
    parser.add_argument("--nogui", action="store_true", help="no gui")
    parser.add_argument("--print-obs", action="store_true", help="print obs")
    parser.add_argument("--draw-obs", action="store_true", help="draw obs")
    args = parser.parse_args()

    env = PickFromPileEnv(gui=not args.nogui)
    obs = env.reset()
    if args.print_obs:
        pprint.pprint(obs)

    if args.model is None:
        agent = None
    else:
        agent = DqnAgent(env=env, model=args.model)
        agent.build(training=False)

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
                ee_pose = obs["ee_pose_init"]
                pp.draw_pose((ee_pose[:3], ee_pose[3:]))
                grasp_flags = obs["grasp_flags_init"]
                object_labels = obs["object_labels_init"]
                object_poses = obs["object_poses_init"]

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

                class_id = env.CLASS_IDS[
                    np.argwhere(object_labels[grasp_flags == 1][0])[0, 0]
                ]
                visual_file = mercury.datasets.ycb.get_visual_file(
                    class_id=class_id
                )
                for pose in obs["past_grasped_object_poses"]:
                    if (pose == 0).all():
                        continue
                    mercury.pybullet.create_mesh_body(
                        visual_file=visual_file,
                        texture=False,
                        rgba_color=(0.5, 0.5, 0.5, 0.5),
                        position=pose[:3],
                        quaternion=pose[3:],
                        mesh_scale=(1.05, 1.05, 1.05),
                    )
                while True:
                    pp.step_simulation()

            obs = env.reset()
        else:
            obs = transition.observation


if __name__ == "__main__":
    main()
