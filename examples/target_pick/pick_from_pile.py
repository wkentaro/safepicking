#!/usr/bin/env python

import argparse
import json
import sys
import time

from loguru import logger
import numpy as np
import path
import pybullet as p
import pybullet_planning as pp

import mercury

import utils


here = path.Path(__file__).abspath().parent


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("export_file", type=path.Path, help="export file")
    parser.add_argument(
        "--planner",
        choices=["RRTConnect", "Naive"],
        required=True,
        help="planner",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--nogui", action="store_true", help="no gui")
    args = parser.parse_args()

    random_state = np.random.RandomState(args.seed)

    plane = utils.init_world(camera_distance=1.5, use_gui=not args.nogui)

    ri = mercury.pybullet.PandaRobotInterface(
        suction_max_force=10, planner=args.planner
    )
    c_cam_to_ee = mercury.geometry.Coordinate()
    c_cam_to_ee.translate([0, -0.05, -0.1])
    ri.add_camera(
        pose=c_cam_to_ee.pose,
        height=240,
        width=320,
    )

    data = np.load(args.export_file)

    is_occluded = data["visibility"] < 0.9
    if is_occluded.sum() == 0:
        logger.warning("no occluded object is found")
        sys.exit(1)
    else:
        target_index = random_state.choice(np.where(is_occluded)[0])

    pile_center = [0.5, 0, 0]

    num_instances = len(data["class_id"])
    object_ids = []
    for i in range(num_instances):
        class_id = data["class_id"][i]
        position = data["position"][i]
        quaternion = data["quaternion"][i]

        position += pile_center

        visual_file = mercury.datasets.ycb.get_visual_file(class_id=class_id)
        collision_file = mercury.pybullet.get_collision_file(visual_file)

        if i == target_index:
            texture = False
            rgba_color = (1, 0, 0)
        else:
            texture = False
            rgba_color = (0.5, 0.5, 0.5)

        class_name = mercury.datasets.ycb.class_names[class_id]
        visibility = data["visibility"][i]
        logger.info(
            f"class_id={class_id:02d}, "
            f"class_name={class_name}, "
            f"visibility={visibility:.1%}"
        )

        with pp.LockRenderer():
            object_id = mercury.pybullet.create_mesh_body(
                visual_file=visual_file,
                collision_file=collision_file,
                mass=mercury.datasets.ycb.masses[class_id],
                position=position,
                quaternion=quaternion,
                rgba_color=rgba_color,
                texture=texture,
            )
        object_ids.append(object_id)

    c = mercury.geometry.Coordinate(*ri.get_pose("camera_link"))
    c.position = pp.get_pose(object_ids[target_index])[0]
    c.position[2] += 0.5
    while True:
        j = ri.solve_ik(c.pose, move_target=ri.robot_model.camera_link)
        js = ri.planj(j)
        for j in js:
            for _ in ri.movej(j):
                p.stepSimulation()
                if not args.nogui:
                    time.sleep(1 / 240)

        rgb, depth, segm = ri.get_camera_image()

        for _ in ri.random_grasp(
            depth,
            segm,
            bg_object_ids=[plane],
            object_ids=object_ids,
            target_object_ids=[object_ids[target_index]],
            random_state=random_state,
            noise=False,
        ):
            p.stepSimulation()
            if not args.nogui:
                time.sleep(1 / 240)
        if (
            ri.gripper.check_grasp()
            and ri.attachments[0].child == object_ids[target_index]
        ):
            break
        else:
            ri.ungrasp()

    i = 0
    velocities = {}
    for _ in ri.move_to_homej(
        bg_object_ids=[plane], object_ids=object_ids, speed=0.001
    ):
        i += 1

        p.stepSimulation()
        ri.step_simulation()
        if not args.nogui:
            time.sleep(1 / 240)

        for object_id in object_ids:
            if object_id == object_ids[target_index]:
                continue
            velocities[object_id] = max(
                velocities.get(object_id, 0),
                np.linalg.norm(pp.get_velocity(object_id)[0]),
            )

    success = ri.gripper.check_grasp()
    if success:
        logger.success("Task is complete")
    else:
        logger.error("Task is failed")

    for object_id in object_ids:
        if object_id == object_ids[target_index]:
            continue
        logger.info(
            f"object_id={object_id}, "
            f"class_id={utils.get_class_id(object_id):02d}, "
            f"velocity={velocities[object_id]:.3f}"
        )
    logger.info(f"sum_of_velocities: {sum(velocities.values()):.3f}")

    scene_id = args.export_file.stem
    data = dict(
        planner=args.planner,
        scene_id=scene_id,
        seed=args.seed,
        success=success,
        velocities=list(velocities.values()),
        sum_of_velocities=sum(velocities.values()),
    )

    json_file = here / f"logs/{args.planner}/{scene_id}/{args.seed}.json"
    json_file.parent.makedirs_p()
    with open(json_file, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"saved to: {json_file}")


if __name__ == "__main__":
    main()
