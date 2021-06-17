#!/usr/bin/env python

import pickle
import pprint

from loguru import logger
import numpy as np
import path
import pybullet as p
import pybullet_planning as pp

import mercury

import common_utils
from env import Env


home = path.Path("~").expanduser()


def view_pkl_file(pkl_file):
    pp.reset_simulation()
    p.setGravity(0, 0, -9.8)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=90,
        cameraPitch=-60,
        cameraTargetPosition=(0, 0, 0),
    )
    pp.load_pybullet("plane.urdf")
    ri = mercury.pybullet.PandaRobotInterface()

    bin = mercury.pybullet.create_bin(*Env.BIN_EXTENTS)
    pp.set_pose(bin, Env.BIN_POSE)

    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    if np.isnan(data["length"]):
        logger.error("\n" + pprint.pformat(data))
    else:
        logger.success("\n" + pprint.pformat(data))

    object_ids = []
    for fg_flag, class_id, pose in zip(
        data["object_fg_flags"], data["object_classes"], data["object_poses"]
    ):
        visual_file = mercury.datasets.ycb.get_visual_file(class_id=class_id)
        object_id = mercury.pybullet.create_mesh_body(
            visual_file=visual_file,
            collision_file=mercury.pybullet.get_collision_file(visual_file),
            position=pose[:3],
            quaternion=pose[3:],
            mass=mercury.datasets.ycb.masses[class_id],
        )
        object_ids.append(object_id)
        if fg_flag:
            pp.draw_aabb(pp.get_aabb(object_id), color=(1, 0, 0, 1), width=2)

            object_ids.append(
                mercury.pybullet.duplicate(
                    object_id,
                    texture=True,
                    collision=False,
                    rgba_color=(0, 1, 0, 0.5),
                    position=data["reorient_pose"][:3],
                    quaternion=data["reorient_pose"][3:],
                )
            )

    j = ri.solve_ik((data["grasp_pose"][:3], data["grasp_pose"][3:]))
    ri.setj(j)

    common_utils.pause(enabled=True)


def main():
    pp.connect(use_gui=True)
    pp.add_data_path()

    root_dir = path.Path(
        "/home/wkentaro/data/mercury/reorient/class_2_3_5_11_12_15/"
    )
    for pkl_file in sorted(root_dir.listdir()):
        view_pkl_file(pkl_file=pkl_file)


if __name__ == "__main__":
    main()
