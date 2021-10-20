import time

import numpy as np
import pybullet as p
import pybullet_planning as pp

import mercury


def draw_points(points, colors, size=1):
    points = np.asarray(points)
    colors = np.asarray(colors)

    if colors.dtype == np.uint8:
        colors = colors.astype(np.float32) / 255

    assert points.shape[-1] == 3
    assert colors.shape[-1] == 3

    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)

    mask = ~np.isnan(points).any(axis=1)
    points = points[mask]
    colors = colors[mask]

    N = len(points)
    assert len(colors) == N

    MAX_NUM_POINTS = 130000
    if N > MAX_NUM_POINTS:
        i = np.random.permutation(N)[:MAX_NUM_POINTS]
    else:
        i = Ellipsis
    return p.addUserDebugPoints(points[i], colors[i], pointSize=size)


def annotate_pose(obj):
    while True:
        events = p.getKeyboardEvents()

        dp = 0.0001
        dr = 0.001

        sign = 1
        if events.get(65306) == p.KEY_IS_DOWN:  # SHIFT
            sign = -1
        if events.get(65307) == p.KEY_IS_DOWN:  # CTRL
            dp *= 10
            dr *= 10

        c = mercury.geometry.Coordinate(*pp.get_pose(obj))
        if events.get(ord("k")) == p.KEY_IS_DOWN:
            c.translate([dp, 0, 0], wrt="world")
        elif events.get(ord("j")) == p.KEY_IS_DOWN:
            c.translate([-dp, 0, 0], wrt="world")
        elif events.get(ord("l")) == p.KEY_IS_DOWN:
            c.translate([0, -dp, 0], wrt="world")
        elif events.get(ord("h")) == p.KEY_IS_DOWN:
            c.translate([0, dp, 0], wrt="world")
        elif events.get(ord("i")) == p.KEY_IS_DOWN:
            c.translate([0, 0, dp], wrt="world")
        elif events.get(ord("m")) == p.KEY_IS_DOWN:
            c.translate([0, 0, -dp], wrt="world")
        elif events.get(ord("1")) == p.KEY_IS_DOWN:
            c.rotate([dr * sign, 0, 0], wrt="world")
        elif events.get(ord("2")) == p.KEY_IS_DOWN:
            c.rotate([0, dr * sign, 0], wrt="world")
        elif events.get(ord("3")) == p.KEY_IS_DOWN:
            c.rotate([0, 0, dr * sign], wrt="world")
        elif events.get(ord("c")) == p.KEY_WAS_RELEASED:
            camera = pp.get_camera()
            print(
                f"""
p.resetDebugVisualizerCamera(
    cameraYaw={camera.yaw},
    cameraPitch={camera.pitch},
    cameraDistance={camera.dist},
    cameraTargetPosition={camera.target},
)
"""
            )
        elif events.get(ord("p")) == p.KEY_WAS_RELEASED:
            pose = pp.get_pose(obj)
            print(f"pp.set_pose(obj, {pose})")
        pp.set_pose(obj, c.pose)

        time.sleep(1 / 240)
