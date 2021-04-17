#!/usr/bin/env python

import time

import imgviz
import numpy as np
import path
import pybullet

import mercury

from create_bin import create_bin


def create_clutter(T_base2world=None, class_ids=None, num_instances=None):
    x = (-0.2, 0.2)
    y = (-0.2, 0.2)
    z = 0.5

    bin_unique_id = create_bin(
        X=x[1] - x[0],
        Y=y[1] - y[0],
        Z=z / 2,
    )
    if T_base2world is not None:
        position, quaternion = pybullet.getBasePositionAndOrientation(
            bin_unique_id
        )
        coord = mercury.geometry.Coordinate(
            position=position, quaternion=quaternion
        )
        coord.transform(T_base2world, wrt="world")
        pybullet.resetBasePositionAndOrientation(
            bin_unique_id, coord.position, coord.quaternion
        )

    if class_ids is None:
        class_ids = np.arange(1, 22)
    if num_instances is None:
        num_instances = 44

    unique_id_to_class_id = {}
    class_ids = np.random.choice(class_ids, num_instances).tolist()
    while class_ids:
        class_id = class_ids.pop()

        position = np.random.uniform([x[0], y[0], z], [x[1], y[1], z])
        quaternion = np.random.random((4,))
        quaternion /= np.linalg.norm(quaternion)

        coord = mercury.geometry.Coordinate(position, quaternion=quaternion)
        if T_base2world is not None:
            coord.transform(T_base2world, wrt="world")

        visual_file = mercury.datasets.ycb.get_visual_file(class_id=class_id)
        collision_file = mercury.pybullet.get_collision_file(visual_file)
        unique_id = mercury.pybullet.create_mesh_body(
            visual_file=collision_file,
            collision_file=collision_file,
            mass=0.1,
            position=coord.position,
            quaternion=coord.quaternion,
            rgba_color=imgviz.label_colormap()[class_id] / 255,
        )

        for _ in range(1000):
            pybullet.stepSimulation()
            if np.linalg.norm(pybullet.getBaseVelocity(unique_id)[0]) < 1e-12:
                break

        aabb_min, aabb_max = pybullet.getAABB(bin_unique_id)

        position, _ = pybullet.getBasePositionAndOrientation(unique_id)
        if not (
            (aabb_min[0] < position[0] < aabb_max[0])
            and (aabb_min[1] < position[1] < aabb_max[1])
        ):
            pybullet.removeBody(unique_id)
            class_ids.append(class_id)
        else:
            unique_id_to_class_id[unique_id] = class_id

    for _ in range(250):
        position, quaternion = pybullet.getBasePositionAndOrientation(
            bin_unique_id
        )
        coord = mercury.geometry.Coordinate(
            position=position, quaternion=quaternion
        )
        coord.translate([0, 0, -0.001], wrt="world")
        pybullet.resetBasePositionAndOrientation(
            bin_unique_id, coord.position, coord.quaternion
        )
        for _ in range(100):
            pybullet.stepSimulation()
            if all(
                np.linalg.norm(pybullet.getBaseVelocity(unique_id)[0]) < 1e-12
                for unique_id in mercury.pybullet.get_body_unique_ids()
            ):
                break
    pybullet.removeBody(bin_unique_id)

    return unique_id_to_class_id


here = path.Path(__file__).abspath().parent


def main():
    mercury.pybullet.init_world()

    pybullet.resetDebugVisualizerCamera(
        cameraDistance=1,
        cameraYaw=-60,
        cameraPitch=-60,
        cameraTargetPosition=(0, 0, 0),
    )

    np.random.seed(1)
    unique_id_to_class_id = create_clutter(
        class_ids=[2, 3, 5, 11, 12, 15, 16], num_instances=8
    )

    logs_dir = here / "logs"
    logs_dir.makedirs_p()

    class_ids = []
    positions = []
    quaternions = []
    for unique_id, class_id in unique_id_to_class_id.items():
        position, quaternion = pybullet.getBasePositionAndOrientation(
            unique_id
        )
        class_ids.append(class_id)
        positions.append(position)
        quaternions.append(quaternion)
    np.savez_compressed(
        logs_dir / "pile.npz",
        class_ids=class_ids,
        positions=positions,
        quaternions=quaternions,
    )

    while True:
        time.sleep(1 / 240)
        pybullet.stepSimulation()


if __name__ == "__main__":
    main()
