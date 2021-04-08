import argparse
import itertools
import time

import imgviz
import numpy as np
import pybullet as p
import pybullet_planning as pp

import mercury


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pause", action="store_true", help="pause")
    parser.add_argument(
        "--enable-visual", action="store_true", help="enable visual"
    )
    parser.add_argument(
        "--camera-config", type=int, default=0, help="camera config"
    )
    parser.add_argument("--imshow", action="store_true", help="imshow")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    return parser


def init_world():
    pp.connect()
    pp.add_data_path()
    p.setGravity(0, 0, -9.8)

    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=90,
        cameraPitch=-60,
        cameraTargetPosition=(0, 0, 0),
    )

    plane = p.loadURDF("plane.urdf")
    return plane


def load_pile(base_pose, npz_file, mass=None, enable_visual=False):
    data = np.load(npz_file)
    object_ids = []
    for class_id, position, quaternion in zip(
        data["class_ids"], data["positions"], data["quaternions"]
    ):
        coord = mercury.geometry.Coordinate(
            position=position,
            quaternion=quaternion,
        )
        coord.transform(
            mercury.geometry.transformation_matrix(*base_pose), wrt="world"
        )

        visual_file = mercury.datasets.ycb.get_visual_file(class_id)
        collision_file = mercury.pybullet.get_collision_file(visual_file)
        if enable_visual:
            rgba_color = None
        else:
            visual_file = collision_file
            rgba_color = imgviz.label_colormap()[class_id] / 255
        mass_actual = mercury.datasets.ycb.masses[class_id]
        object_id = mercury.pybullet.create_mesh_body(
            visual_file=visual_file,
            rgba_color=rgba_color,
            collision_file=collision_file,
            mass=mass_actual if mass is None else mass,
            position=coord.position,
            quaternion=coord.quaternion,
        )
        object_ids.append(object_id)
    return object_ids


def get_camera_pose(camera_config):
    c_cam_to_ee = mercury.geometry.Coordinate()

    if camera_config == 0:
        c_cam_to_ee.translate([0, -0.05, -0.1])
    elif camera_config == 1:
        c_cam_to_ee.rotate([np.deg2rad(-15), 0, 0])
        c_cam_to_ee.translate([0, -0.08, -0.2])
    elif camera_config == 2:
        c_cam_to_ee.rotate([np.deg2rad(-15), 0, 0])
        c_cam_to_ee.translate([0, -0.08, -0.35])
    else:
        raise ValueError

    return c_cam_to_ee.pose


def pause(enabled):
    if enabled:
        print("Please press 'n' to start")
        while True:
            if ord("n") in p.getKeyboardEvents():
                break


class StepSimulation:
    def __init__(self, ri, imshow=False):
        self.ri = ri
        self.imshow = imshow

        self.i = 0

    def __call__(self):
        p.stepSimulation()
        self.ri.step_simulation()
        if self.imshow and self.i % 8 == 0:
            rgb, depth, _ = self.ri.get_camera_image()
            depth[(depth < 0.3) | (depth > 2)] = np.nan
            tiled = imgviz.tile(
                [
                    rgb,
                    imgviz.depth2rgb(depth, min_value=0.3, max_value=0.6),
                ],
                border=(255, 255, 255),
            )
            imgviz.io.cv_imshow(tiled)
            imgviz.io.cv_waitkey(1)
        time.sleep(1 / 240 / 2)
        self.i += 1


def get_canonical_quaternion(class_id):
    if class_id == 15:
        quaternion = mercury.geometry.quaternion_from_euler(
            [np.deg2rad(90), 0, 0]
        )
    else:
        quaternion = [0, 0, 0, 1]
    return quaternion


def place_to_regrasp(
    ri, regrasp_aabb, bg_object_ids, object_ids, step_simulation
):
    n_trial = 20

    object_id = ri.attachments[0].child
    visual_shape_data = p.getVisualShapeData(object_id)
    class_name = visual_shape_data[0][4].decode().split("/")[-2]
    class_id = mercury.datasets.ycb.class_names.tolist().index(class_name)

    for i in itertools.count():
        with pp.LockRenderer():
            with pp.WorldSaver():
                quaternion = get_canonical_quaternion(class_id=class_id)
                if i >= n_trial:
                    c = mercury.geometry.Coordinate(quaternion=quaternion)
                    euler = [
                        [np.deg2rad(90), 0, 0],
                        [np.deg2rad(-90), 0, 0],
                        [0, np.deg2rad(90), 0],
                        [0, np.deg2rad(-90), 0],
                    ][np.random.randint(0, 4)]
                    c.rotate(euler, wrt="world")
                    quaternion = c.quaternion

                pp.set_pose(object_id, ([0, 0, 0], quaternion))
                aabb = pp.get_aabb(object_id)
                position = np.random.uniform(*regrasp_aabb)
                position[2] -= aabb[0][2]
                c = mercury.geometry.Coordinate(position, quaternion)
                c.rotate([0, 0, np.random.uniform(-np.pi, np.pi)], wrt="world")
                obj_af_to_world = c.pose

                pp.set_pose(object_id, c.pose)

        with ri.enabling_attachments():
            j = ri.solve_ik(
                obj_af_to_world,
                move_target=ri.robot_model.attachment_link0,
            )
        if j is None:
            print("j is None")
            continue

        obstacles = bg_object_ids + object_ids
        obstacles.remove(object_id)
        path = ri.planj(
            j,
            obstacles=obstacles,
            attachments=ri.attachments,
        )
        if path is None:
            print("path is None")
            continue

        with pp.LockRenderer():
            with pp.WorldSaver():
                ri.setj(path[-1])
                c = mercury.geometry.Coordinate(*ri.get_pose("tipLink"))
                c.translate([0, 0, -0.05])
                j2 = ri.solve_ik(c.pose, rotation_axis=None)
        if j2 is None:
            print("j2 is None")
            continue

        break
    for _ in (_ for j in path for _ in ri.movej(j)):
        step_simulation()

    for _ in range(240):
        step_simulation()

    ri.ungrasp()

    for _ in range(240):
        step_simulation()

    for _ in ri.movej(j2):
        step_simulation()

    path = None
    while path is None:
        path = ri.planj(
            ri.homej,
            obstacles=bg_object_ids + object_ids,
        )
    for _ in (_ for j in path for _ in ri.movej(j)):
        step_simulation()

    return i < n_trial


def draw_grasped_object(ri):
    if not ri.gripper.grasped_object:
        return
    visual_shape_data = p.getVisualShapeData(ri.gripper.grasped_object)
    visual_file = visual_shape_data[0][4].decode()
    obj = mercury.pybullet.create_mesh_body(
        visual_file=visual_file,
        mass=0.001,
        rgba_color=[0, 1, 0, 0.5],
    )
    obj_to_ee = ri.attachments[0].grasp_pose
    ee_to_world = ri.get_pose("tipLink")
    obj_to_world = pp.multiply(ee_to_world, obj_to_ee)
    pp.set_pose(obj, obj_to_world)
    p.createConstraint(
        parentBodyUniqueId=ri.robot,
        parentLinkIndex=ri.ee,
        childBodyUniqueId=obj,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=obj_to_ee[0],
        parentFrameOrientation=obj_to_ee[1],
        childFramePosition=(0, 0, 0),
        childFrameOrientation=(0, 0, 0),
    )

    time.sleep(1)

    p.removeBody(obj)
