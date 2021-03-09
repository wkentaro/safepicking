import shlex
import subprocess

import numpy as np
import path
import pybullet
import pybullet_data


def init_world(*args, **kwargs):
    pybullet.connect(pybullet.GUI)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    pybullet.loadURDF("plane.urdf")
    pybullet.setGravity(0, 0, -9.8)


def get_body_unique_ids():
    num_bodies = pybullet.getNumBodies()
    unique_ids = [pybullet.getBodyUniqueId(i) for i in range(num_bodies)]
    return unique_ids


def create_mesh_body(
    visual_file,
    collision_file=None,
    position=None,
    quaternion=None,
    mass=0,
    rgba_color=None,
):
    if collision_file is None:
        collision_file = get_collision_file(visual_file)
    if rgba_color is not None and len(rgba_color) == 3:
        rgba_color = [rgba_color[0], rgba_color[1], rgba_color[2], 1]
    visual_shape_id = pybullet.createVisualShape(
        shapeType=pybullet.GEOM_MESH,
        fileName=visual_file,
        visualFramePosition=[0, 0, 0],
        meshScale=[1, 1, 1],
        rgbaColor=rgba_color,
    )
    collision_shape_id = pybullet.createCollisionShape(
        shapeType=pybullet.GEOM_MESH,
        fileName=collision_file,
        collisionFramePosition=[0, 0, 0],
        meshScale=[1, 1, 1],
    )
    unique_id = pybullet.createMultiBody(
        baseMass=mass,
        baseInertialFramePosition=[0, 0, 0],
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=position,
        baseOrientation=quaternion,
        useMaximalCoordinates=False,
    )
    return unique_id


def get_collision_file(visual_file):
    visual_file = path.Path(visual_file)
    collision_file = visual_file.stripext() + ".convex" + visual_file.ext
    if not collision_file.exists():
        cmd = (
            f"testVHACD --input {visual_file} --output {collision_file}"
            " --log /tmp/testVHACD.log --resolution 200000"
        )
        subprocess.check_output(shlex.split(cmd))
    return collision_file


def get_debug_visualizer_image():
    width, height, *_ = pybullet.getDebugVisualizerCamera()
    _, _, rgba, depth, segm = pybullet.getCameraImage(
        width=width, height=height
    )
    rgb = rgba[:, :, :3]
    depth[segm == -1] = np.nan
    return rgb, depth, segm


def get_aabb(unique_id):
    aabb_min, aabb_max = pybullet.getAABB(unique_id)
    return np.array(aabb_min), np.array(aabb_max)


def is_colliding(id1):
    is_colliding = False
    for id2 in get_body_unique_ids():
        if id1 != id2:
            points = pybullet.getClosestPoints(id1, id2, distance=0)
            if points:
                is_colliding = True
    return is_colliding
