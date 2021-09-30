import pickle

import imgviz
import numpy as np
import pybullet as p
import pybullet_planning as pp

import mercury


def init_simulation(camera_distance=1.5):
    pp.add_data_path()
    p.setGravity(0, 0, -9.8)

    p.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=90,
        cameraPitch=-60,
        cameraTargetPosition=(0, 0, 0),
    )

    plane = p.loadURDF("plane.urdf")
    return plane


def load_pile(base_pose, pkl_file, mass=None):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    object_ids = []
    for class_id, position, quaternion in zip(
        data["class_id"], data["position"], data["quaternion"]
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
        mass_actual = mercury.datasets.ycb.masses[class_id]
        with pp.LockRenderer():
            object_id = mercury.pybullet.create_mesh_body(
                visual_file=visual_file,
                collision_file=collision_file,
                mass=mass_actual if mass is None else mass,
                position=coord.position,
                quaternion=coord.quaternion,
                rgba_color=imgviz.label_colormap()[class_id] / 255,
                texture=False,
            )
        object_ids.append(object_id)
    return object_ids


def get_class_id(object_id):
    visual_shape_data = p.getVisualShapeData(object_id)
    class_name = visual_shape_data[0][4].decode().split("/")[-2]
    class_id = mercury.datasets.ycb.class_names.tolist().index(class_name)
    return class_id


def get_canonical_quaternion(class_id):
    c = mercury.geometry.Coordinate()
    if class_id == 2:
        c.rotate([0, 0, np.deg2rad(0)])
    elif class_id == 3:
        c.rotate([0, 0, np.deg2rad(5)])
    elif class_id == 5:
        c.rotate([0, 0, np.deg2rad(-65)])
    elif class_id == 11:
        c.rotate([0, 0, np.deg2rad(47)])
    elif class_id == 12:
        c.rotate([0, 0, np.deg2rad(90)])
    elif class_id == 15:
        c.rotate([0, np.deg2rad(90), np.deg2rad(90)])
    else:
        pass
    return c.quaternion


def pause(enabled):
    if enabled:
        print("Please press 'n' to start")
        while True:
            events = p.getKeyboardEvents()
            if events.get(ord("n")) == 4:
                break
