#!/usr/bin/env python

import time

import imgviz
import numpy as np
import pybullet

import mercury

from create_bin import create_bin


def bin_packing(object_ids, class_ids, bin_aabb_min, bin_aabb_max, sleep=0):
    for object_id, class_id in zip(object_ids, class_ids):
        position_org, quaternion_org = pybullet.getBasePositionAndOrientation(
            object_id
        )

        if class_id == 15:
            quaternion = mercury.geometry.quaternion_from_euler(
                [np.deg2rad(90), 0, 0]
            )
        else:
            quaternion = [0, 0, 0, 1]
        pybullet.resetBasePositionAndOrientation(
            object_id, position_org, quaternion
        )

        aabb_min, aabb_max = np.array(pybullet.getAABB(object_id))
        position_lt = bin_aabb_min - (aabb_min - position_org)
        position_rb = bin_aabb_max + (aabb_min - position_org)

        z = position_lt[2]
        for x in np.linspace(position_lt[0], position_rb[0]):
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
            for y in np.linspace(position_lt[1], position_rb[1]):
                pybullet.resetBasePositionAndOrientation(
                    object_id, [x, y, z], quaternion
                )
                if 0:
                    time.sleep(0.01)
                if not mercury.pybullet.is_colliding(object_id):
                    break
            else:
                continue
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
            time.sleep(sleep)
            break
        else:
            pybullet.resetBasePositionAndOrientation(
                object_id, position_org, quaternion_org
            )
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
            break


def main():
    mercury.pybullet.init_world()

    pybullet.resetDebugVisualizerCamera(
        cameraDistance=1,
        cameraYaw=90,
        cameraPitch=-60,
        cameraTargetPosition=(0, 0, 0),
    )

    bin_unique_id = create_bin(0.5, 0.5, 0.2)
    bin_aabb_min, bin_aabb_max = np.array(pybullet.getAABB(bin_unique_id))

    bin_aabb_min += 0.01
    bin_aabb_max -= 0.01

    # visual_shape_id = pybullet.createVisualShape(
    #     shapeType=pybullet.GEOM_BOX,
    #     halfExtents=(bin_aabb_max - bin_aabb_min) / 2,
    #     rgbaColor=[1, 0, 0, 0.3],
    # )
    # pybullet.createMultiBody(
    #     baseMass=0,
    #     basePosition=(bin_aabb_max + bin_aabb_min) / 2,
    #     baseOrientation=[0, 0, 0, 1],
    #     baseVisualShapeIndex=visual_shape_id,
    #     baseCollisionShapeIndex=visual_shape_id,
    #     baseInertialFramePosition=[0, 0, 0],
    #     baseInertialFrameOrientation=[0, 0, 0, 1],
    # )

    class_ids = np.arange(1, 22)
    class_ids = class_ids[~np.isin(class_ids, [10, 18])]
    class_ids = np.random.choice(class_ids, 10)
    object_ids = []
    for class_id in class_ids:
        visual_file = mercury.datasets.ycb.get_visual_file(class_id)
        collision_file = mercury.pybullet.get_collision_file(visual_file)

        unique_id = mercury.pybullet.create_mesh_body(
            visual_file=collision_file,
            collision_file=collision_file,
            position=(bin_aabb_min + bin_aabb_max) / 2 + [0, 0, 0.5],
            quaternion=[0, 0, 0, 1],
            mass=0.1,
            rgba_color=imgviz.label_colormap()[class_id] / 255,
        )
        object_ids.append(unique_id)

    bin_packing(object_ids, class_ids, bin_aabb_min, bin_aabb_max, sleep=0.3)

    while True:
        pass


if __name__ == "__main__":
    main()
