import numpy as np
import pybullet as p
import pybullet_planning as pp

import mercury

import _utils


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


def init_place_scene(class_id, face="front"):
    bin = mercury.pybullet.create_bin(0.3, 0.4, 0.2, color=(0.8, 0.8, 0.8, 1))
    c = mercury.geometry.Coordinate([0, 0.6, 0.8])
    c.rotate([np.pi / 2, 0, 0])
    pp.set_pose(bin, c.pose)

    c.quaternion = _utils.get_canonical_quaternion(class_id)
    if face == "front":
        c.rotate([0, 0, np.deg2rad(-90)], wrt="world")
    elif face == "right":
        c.rotate([0, 0, np.deg2rad(0)], wrt="world")
    elif face == "left":
        c.rotate([0, 0, np.deg2rad(180)], wrt="world")
    else:
        raise ValueError
    place_pose = c.pose

    mercury.pybullet.create_mesh_body(
        visual_file=mercury.datasets.ycb.get_visual_file(class_id),
        texture=True,
        position=place_pose[0],
        quaternion=place_pose[1],
        rgba_color=(0.2, 0.8, 0.2, 0.8),
    )

    return [bin], place_pose
