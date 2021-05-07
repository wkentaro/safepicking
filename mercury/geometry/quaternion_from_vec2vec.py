import numpy as np


def quaternion_from_angle_axis(angle, axis):
    half_angle = angle / 2
    sin = np.sin(half_angle)
    w = np.cos(half_angle)
    x = sin * axis[0]
    y = sin * axis[1]
    z = sin * axis[2]
    return np.array([x, y, z, w], dtype=np.float64)


def quaternion_from_vec2vec(v1, v2, flip=True):
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    d = np.dot(v1, v2)

    if d < 0 and flip:
        v2 = -v2
        d = np.dot(v1, v2)

    if d >= 1:
        return np.array([0, 0, 0, 1], dtype=np.float64)
    elif d < (1e-6 - 1):
        axis = np.cross([0, 0, 1], v1)
        if np.linalg.norm(axis) == 0:
            axis = np.cross([0, 1, 0], v1)
        axis = axis / np.linalg.norm(axis)
        return quaternion_from_angle_axis(np.pi, axis)

    s = np.sqrt((1 + d) * 2)
    invs = 1 / s

    v3 = np.cross(v1, v2)

    x, y, z = v3
    x *= invs
    y *= invs
    z *= invs
    w = s / 2
    quaternion = np.array([x, y, z, w], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)
