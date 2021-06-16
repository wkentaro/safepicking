import numpy as np
import pybullet as p
import pybullet_planning as pp

import common_utils
import mercury


def init_place_scene(class_id):
    if class_id == 2:
        container, place_pose = init_place_scene_shelf_front(class_id=class_id)
    else:
        container, place_pose = init_place_scene_simple(class_id=class_id)
    mercury.pybullet.create_mesh_body(
        visual_file=mercury.datasets.ycb.get_visual_file(class_id),
        texture=True,
        position=place_pose[0],
        quaternion=place_pose[1],
        rgba_color=(0.2, 0.8, 0.2, 0.8),
    )
    return container, place_pose


def init_place_scene_simple(class_id):
    bin = mercury.pybullet.create_bin(0.3, 0.4, 0.2, color=(0.8, 0.8, 0.8, 1))
    c = mercury.geometry.Coordinate([0, 0.6, 0.8])
    c.rotate([np.pi / 2, 0, 0])
    pp.set_pose(bin, c.pose)

    c.quaternion = common_utils.get_canonical_quaternion(class_id)
    c.rotate([0, 0, np.deg2rad(-90)], wrt="world")
    place_pose = c.pose
    return [bin], place_pose


def init_place_scene_shelf_front(class_id):
    assert class_id == 2

    lock_renderer = pp.LockRenderer()

    W = 0.55
    H = 0.3
    D = 0.3
    create = [True, True, True, False, True]
    bin_middle = mercury.pybullet.create_bin(
        H, W, D, color=(0.8, 0.8, 0.8, 1), create=create
    )
    pp.set_pose(bin_middle, ((0, 0, 0), (0, 0, 0, 1)))
    bin_top = mercury.pybullet.create_bin(
        H, W, D, color=(0.8, 0.8, 0.8, 1), create=create
    )
    pp.set_pose(bin_top, ([0.3, 0, 0], [0, 0, 0, 1]))
    bin_bottom = mercury.pybullet.create_bin(
        H, W, D, color=(0.8, 0.8, 0.8, 1), create=create
    )
    pp.set_pose(bin_bottom, ([-0.3, 0, 0], [0, 0, 0, 1]))

    shelf = [bin_middle, bin_top, bin_bottom]

    c = mercury.geometry.Coordinate()
    c.rotate([np.deg2rad(90), 0, np.deg2rad(90)])
    c.translate([0, 0.6, 0.45], wrt="world")

    def set_pose(objs, pose):
        obj_base_af_to_world = pose
        obj_base_to_world = pp.get_pose(objs[0])
        for obj in objs:
            obj_to_world = pp.get_pose(obj)
            obj_to_obj_base = pp.multiply(
                pp.invert(obj_base_to_world), obj_to_world
            )
            pp.set_pose(
                obj, pp.multiply(obj_base_af_to_world, obj_to_obj_base)
            )

    set_pose(shelf, c.pose)

    visual_file = mercury.datasets.ycb.get_visual_file(class_id=2)

    positions = [
        [-0.18, 0.7, 0.72],
        [-0.18, 0.63, 0.72],
        [-0.18, 0.56, 0.72],
        [0, 0.7, 0.72],
        [0, 0.63, 0.72],
        [0, 0.56, 0.72],
        [0.18, 0.7, 0.72],
        [0.18, 0.63, 0.72],
        [0.18, 0.56, 0.72],
    ]
    i_place_pose = 8

    for i, position in enumerate(positions):
        c = mercury.geometry.Coordinate(
            position, common_utils.get_canonical_quaternion(2)
        )
        c.rotate([0, 0, np.deg2rad(-90)], wrt="world")
        if i == i_place_pose:
            place_pose = c.pose
        else:
            mercury.pybullet.create_mesh_body(
                visual_file=visual_file,
                collision_file=mercury.pybullet.get_collision_file(
                    visual_file
                ),
                position=c.position,
                quaternion=c.quaternion,
            )
    lock_renderer.restore()
    return shelf, place_pose


def main():
    pp.connect()
    pp.add_data_path()
    pp.load_pybullet("plane.urdf")

    p.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=0,
        cameraPitch=-10,
        cameraTargetPosition=(0, 0.5, 0.5),
    )

    pp.draw_pose(((0, 0, 0), (0, 0, 0, 1)), width=3)

    init_place_scene_shelf_front(class_id=2)

    while True:
        pass


if __name__ == "__main__":
    main()
