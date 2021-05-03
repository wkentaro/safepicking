#!/usr/bin/env python

import argparse
import sys

from loguru import logger
import numpy as np
import path
import pybullet as p
import pybullet_planning as pp

import mercury

import common_utils


home = path.Path("~").expanduser()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--nogui", action="store_true", help="no gui")
    args = parser.parse_args()

    export_dir = home / "data/mercury/pile_generation"
    export_file = export_dir / f"{args.seed:08d}.npz"

    if export_file.exists():
        logger.warning(f"File already exists: {export_file}")
        sys.exit(0)

    pp.connect(use_gui=not args.nogui)
    common_utils.init_simulation(camera_distance=1)

    object_ids = common_utils.create_pile(
        class_ids=[2, 3, 5, 11, 12, 15, 16],
        num_instances=8,
        random_state=np.random.RandomState(args.seed),
    )

    # near orthographic rendering
    T_camera_to_world = mercury.geometry.look_at(
        eye=[0, 0, 10000], target=[0, 0, 0]
    )
    rgb, depth, segm = mercury.pybullet.get_camera_image(
        T_camera_to_world,
        fovy=np.deg2rad(0.005),
        height=480,
        width=480,
        near=9999.7,
        far=10001,
    )

    visibilities = {}
    for object_id in object_ids:
        stashed_object_ids = object_ids.copy()
        stashed_object_ids.remove(object_id)
        with mercury.pybullet.stash_objects(stashed_object_ids):
            mask_full = (
                mercury.pybullet.get_camera_image(
                    T_camera_to_world,
                    fovy=np.deg2rad(0.005),
                    height=480,
                    width=480,
                    near=9999.7,
                    far=10001,
                )[2]
                == object_id
            )
        mask_visible = segm == object_id
        intersection = mask_visible & mask_full
        visibility = intersection.sum() / mask_full.sum()
        visibilities[object_id] = visibility

    data = dict(class_id=[], position=[], quaternion=[], visibility=[])
    for object_id in object_ids:
        class_id = int(p.getUserData(p.getUserDataId(object_id, "class_id")))
        pose = pp.get_pose(object_id)
        data["class_id"].append(class_id)
        data["position"].append(pose[0])
        data["quaternion"].append(pose[1])
        data["visibility"].append(visibilities[object_id])

    export_file.parent.makedirs_p()
    np.savez_compressed(export_file, **data)
    logger.success(f"Saved to: {export_file}")


if __name__ == "__main__":
    main()
