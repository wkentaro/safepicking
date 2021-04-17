#!/usr/bin/env python

import time

import imgviz
import numpy as np
import path
import pybullet

import mercury

from create_pile import get_cad_file


here = path.Path(__file__).abspath().parent


def main():
    mercury.pybullet.init_world()

    pybullet.resetDebugVisualizerCamera(
        cameraDistance=1,
        cameraYaw=-60,
        cameraPitch=-60,
        cameraTargetPosition=(0, 0, 0),
    )

    data = np.load(here / "data/pile.npz")
    class_ids = data["class_ids"]
    positions = data["positions"]
    quaternions = data["quaternions"]

    unique_ids = []
    for (
        class_id,
        position,
        quaternion,
    ) in zip(class_ids, positions, quaternions):
        visual_file = get_cad_file(class_id=class_id)
        collision_file = mercury.pybullet.get_collision_file(visual_file)
        unique_id = mercury.pybullet.create_mesh_body(
            visual_file=collision_file,
            collision_file=collision_file,
            position=position,
            quaternion=quaternion,
            mass=0.1,
            rgba_color=imgviz.label_colormap()[class_id] / 255,
        )
        unique_ids.append(unique_id)

    time.sleep(1)

    save_dir = here / "data/disturbed_pile.v2"
    save_dir.makedirs_p()
    for i_scene in range(1000):
        for unique_id, position, quaternion in zip(
            unique_ids, positions, quaternions
        ):
            pybullet.resetBasePositionAndOrientation(
                unique_id, position, quaternion
            )

        for i in [5]:
            unique_id = unique_ids[i]
            position = positions[i]
            quaternion = quaternions[i]

            coord = mercury.geometry.Coordinate(
                position=position, quaternion=quaternion
            )
            coord.translate(np.random.uniform([-0.04] * 3, [0.04] * 3))
            coord.rotate(
                np.random.uniform([-np.deg2rad(30)] * 3, [np.deg2rad(30)] * 3)
            )
            pybullet.resetBasePositionAndOrientation(
                unique_id, coord.position, coord.quaternion
            )

        positions_disturbed = []
        quaternions_disturbed = []
        for unique_id in unique_ids:
            position, quaternion = pybullet.getBasePositionAndOrientation(
                unique_id
            )
            positions_disturbed.append(position)
            quaternions_disturbed.append(quaternion)

        # 0.25s
        for _ in range(60):
            pybullet.stepSimulation()

        delta_positions = []
        for unique_id, position_org in zip(unique_ids, positions):
            (
                position_now,
                quaternion_now,
            ) = pybullet.getBasePositionAndOrientation(unique_id)
            delta_position = np.linalg.norm(position_now - position_org)
            delta_positions.append(delta_position)

        print(delta_positions[5])

        npz_file = save_dir / f"{i_scene:04d}.npz"
        np.savez_compressed(
            npz_file,
            class_ids=class_ids,
            positions=positions_disturbed,
            quaternions=quaternions_disturbed,
            delta_positions=delta_positions,
        )
        print(f"==> Saved to: {npz_file}")


if __name__ == "__main__":
    main()
