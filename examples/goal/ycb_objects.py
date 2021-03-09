#!/usr/bin/env python

from create_pile import get_cad_file

import numpy as np
import pybullet

import mercury


def main():
    mercury.pybullet.init_world()

    pybullet.resetDebugVisualizerCamera(
        cameraDistance=1,
        cameraYaw=0,
        cameraPitch=-40,
        cameraTargetPosition=(1.2, 0, 0),
    )

    base_position = np.zeros((3,))
    for i, class_id in enumerate(range(1, 22)):
        visual_file = get_cad_file(class_id)
        unique_id = mercury.pybullet.create_mesh_body(visual_file=visual_file)
        aabb_min, aabb_max = mercury.pybullet.get_aabb(unique_id)
        _, quaternion = pybullet.getBasePositionAndOrientation(unique_id)
        position = base_position - aabb_min
        pybullet.resetBasePositionAndOrientation(
            unique_id, position, quaternion
        )
        aabb_extents = aabb_max - aabb_min
        base_position += [aabb_extents[0], 0, 0]

        pybullet.addUserDebugText(
            str(class_id),
            position + [0, 0, 0.2],
            textColorRGB=[0, 0, 0],
            textSize=1,
        )

    while True:
        pass


if __name__ == "__main__":
    main()
