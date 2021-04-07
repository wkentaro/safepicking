import numpy as np
import pybullet as p


def create_bin(X, Y, Z):
    origin = [0, 0, 0]

    def get_parts(origin, X, Y, Z, T=0.01):
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

    extents, positions = get_parts(origin, X, Y, Z)

    color = [150, 111, 51, 255]
    halfExtents = np.array(extents) / 2
    shapeTypes = [p.GEOM_BOX] * len(extents)
    rgbaColors = np.array([color] * len(extents)) / 255
    visual_shape_id = p.createVisualShapeArray(
        shapeTypes=shapeTypes,
        halfExtents=halfExtents,
        visualFramePositions=positions,
        rgbaColors=rgbaColors,
    )
    collision_shape_id = p.createCollisionShapeArray(
        shapeTypes=shapeTypes,
        halfExtents=halfExtents,
        collisionFramePositions=positions,
    )

    position = [0, 0, Z / 2]
    quaternion = [0, 0, 0, 1]
    unique_id = p.createMultiBody(
        baseMass=0,
        basePosition=position,
        baseOrientation=quaternion,
        baseVisualShapeIndex=visual_shape_id,
        baseCollisionShapeIndex=collision_shape_id,
        baseInertialFramePosition=[0, 0, 0],
        baseInertialFrameOrientation=[0, 0, 0, 1],
    )
    return unique_id
