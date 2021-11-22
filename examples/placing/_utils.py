import numpy as np
import pybullet as p


def create_shelf(X, Y, Z, N=3):
    T = 0.01
    color = (0.8, 0.8, 0.8, 1)

    def get_parts(origin, X, Y, Z, T):
        extents = np.array(
            [
                [X, Y, T],
                [X, Y, T],
                [X, T, Z],
                [X, T, Z],
                # [T, Y, Z],
                [T, Y, Z],
            ]
        )
        positions = (
            np.array(
                [
                    [0, 0, Z / 2],
                    [0, 0, -Z / 2],
                    [0, Y / 2, 0],
                    [0, -Y / 2, 0],
                    # [X / 2, 0, 0],
                    [-X / 2, 0, 0],
                ]
            )
            + origin
        )
        return extents, positions

    extents = []
    positions = []
    for i in range(N):
        origin = [0, 0, T + Z / 2 + i * (T + Z)]
        parts = get_parts(origin, X, Y, Z, T)
        extents.extend(parts[0])
        positions.extend(parts[1])

    halfExtents = np.array(extents) / 2
    shapeTypes = [p.GEOM_BOX] * len(extents)
    rgbaColors = [color] * len(extents)
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

    position = [0, 0, 0]
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
