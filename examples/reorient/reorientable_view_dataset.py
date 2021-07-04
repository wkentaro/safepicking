#!/usr/bin/env python

import pickle

import imgviz
from loguru import logger
import numpy as np
import path
import pybullet_planning as pp

import mercury


home = path.Path("~").expanduser()


def main():
    pp.connect()
    pp.add_data_path()

    pp.set_camera_pose([1, 0, 1])

    root_dir = home / "data/mercury/reorient/reorientable"
    pkl_files = sorted(root_dir.walk("*.pkl"))

    while True:
        pp.reset_simulation()
        pp.enable_gravity()
        pp.load_pybullet("plane.urdf")
        mercury.pybullet.PandaRobotInterface()

        pkl_file = np.random.choice(pkl_files)
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        heightmap = data["pointmap"][:, :, 2]
        imgviz.io.cv_imshow(imgviz.depth2rgb(heightmap))
        imgviz.io.cv_waitkey(100)

        target_obj = None
        for object_fg_flag, object_class, object_pose in zip(
            data["object_fg_flags"],
            data["object_classes"],
            data["object_poses"],
        ):
            visual_file = mercury.datasets.ycb.get_visual_file(
                class_id=object_class
            )
            obj = mercury.pybullet.create_mesh_body(
                visual_file=visual_file,
                collision_file=mercury.pybullet.get_collision_file(
                    visual_file
                ),
                position=object_pose[:3],
                quaternion=object_pose[3:],
                mass=mercury.datasets.ycb.masses[object_class],
            )
            if object_fg_flag:
                target_obj = obj

        pp.draw_aabb(pp.get_aabb(target_obj), color=(1, 0, 0, 1), width=2)

        ee_to_obj = np.hsplit(data["grasp_pose_wrt_obj"], [3])
        obj_af_to_world = np.hsplit(data["reorient_pose"], [3])

        logger.info(pkl_file)
        print(
            {
                key: data[key]
                for key in ["graspable", "placable", "reorientable"]
            }
        )

        mercury.pybullet.duplicate(
            target_obj,
            collision=False,
            position=obj_af_to_world[0],
            quaternion=obj_af_to_world[1],
            mass=0,
        )

        pp.draw_pose(ee_to_obj, parent=target_obj, length=0.05, width=3)
        mercury.pybullet.pause()


if __name__ == "__main__":
    main()
