#!/usr/bin/env python

import numpy as np
import pybullet

import mercury


def get_container(origin, X, Y, Z, T=0.01):
    extents = [[X, Y, T], [X, T, Z], [X, T, Z], [T, Y, Z], [T, Y, Z]]
    positions = [
        [0, 0, -Z / 2],
        [0, Y / 2, 0],
        [0, -Y / 2, 0],
        [X / 2, 0, 0],
        [-X / 2, 0, 0],
    ]

    positions += np.array(origin)

    return extents, positions


def create_bin(X, Y, Z):
    origin = [0, 0, 0]
    extents, positions = get_container(origin, X, Y, Z)

    color = [150, 111, 51, 255]
    halfExtents = np.array(extents) / 2
    shapeTypes = [pybullet.GEOM_BOX] * len(extents)
    rgbaColors = np.array([color] * len(extents)) / 255
    visual_shape_id = pybullet.createVisualShapeArray(
        shapeTypes=shapeTypes,
        halfExtents=halfExtents,
        visualFramePositions=positions,
        rgbaColors=rgbaColors,
    )
    collision_shape_id = pybullet.createCollisionShapeArray(
        shapeTypes=shapeTypes,
        halfExtents=halfExtents,
        collisionFramePositions=positions,
    )

    position = [0, 0, Z / 2]
    quaternion = [0, 0, 0, 1]
    unique_id = pybullet.createMultiBody(
        baseMass=0,
        basePosition=position,
        baseOrientation=quaternion,
        baseVisualShapeIndex=visual_shape_id,
        baseCollisionShapeIndex=collision_shape_id,
        baseInertialFramePosition=[0, 0, 0],
        baseInertialFrameOrientation=[0, 0, 0, 1],
    )
    return unique_id


def main():
    mercury.pybullet.init_world()

    pybullet.resetDebugVisualizerCamera(
        cameraDistance=1,
        cameraYaw=-60,
        cameraPitch=-20,
        cameraTargetPosition=(0, 0, 0.4),
    )

    create_bin(X=0.4, Y=0.6, Z=0.2)

    while True:
        pybullet.stepSimulation()


if __name__ == "__main__":
    main()
