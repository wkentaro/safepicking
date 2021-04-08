import argparse
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
        time.sleep(1 / 240)
        self.i += 1
