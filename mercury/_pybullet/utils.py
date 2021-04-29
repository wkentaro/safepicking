import contextlib
import itertools
import shlex
import subprocess
import time

import numpy as np
import path
import pybullet
import pybullet_data
import pybullet_planning

from .. import geometry


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
    visual_file=None,
    collision_file=None,
    position=None,
    quaternion=None,
    mass=0,
    rgba_color=None,
    texture=True,
):
    if rgba_color is not None and len(rgba_color) == 3:
        rgba_color = [rgba_color[0], rgba_color[1], rgba_color[2], 1]
    if visual_file is not None:
        visual_shape_id = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_MESH,
            fileName=visual_file,
            visualFramePosition=[0, 0, 0],
            meshScale=[1, 1, 1],
            rgbaColor=rgba_color,
        )
    else:
        visual_shape_id = -1
    if collision_file is not None:
        collision_shape_id = pybullet.createCollisionShape(
            shapeType=pybullet.GEOM_MESH,
            fileName=collision_file,
            collisionFramePosition=[0, 0, 0],
            meshScale=[1, 1, 1],
        )
    else:
        collision_shape_id = -1
    unique_id = pybullet.createMultiBody(
        baseMass=mass,
        baseInertialFramePosition=[0, 0, 0],
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=position,
        baseOrientation=quaternion,
        useMaximalCoordinates=False,
    )
    if not texture:
        pybullet.changeVisualShape(unique_id, -1, textureUniqueId=-1)
    if collision_file:
        pybullet.addUserData(unique_id, "collision_file", collision_file)
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
        width=width,
        height=height,
        renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
    )
    rgb = rgba[:, :, :3]
    depth[segm == -1] = np.nan
    return rgb, depth, segm


def get_aabb(unique_id):
    aabb_min, aabb_max = pybullet.getAABB(unique_id)
    return np.array(aabb_min), np.array(aabb_max)


def is_colliding(id1, ids2=None):
    if ids2 is None:
        ids2 = np.array(get_body_unique_ids())
        ids2 = ids2[ids2 != id1]
    is_colliding = False
    for id2 in ids2:
        points = pybullet.getClosestPoints(id1, id2, distance=0)
        if points:
            is_colliding = True
            break
    return is_colliding


def get_pose(body_id, link_id=-1, parent_body_id=None, parent_link_id=-1):
    self_to_world = pybullet_planning.get_link_pose(body_id, link_id)

    if parent_body_id is None:
        self_to_parent = self_to_world
    else:
        parent_to_world = pybullet_planning.get_link_pose(
            parent_body_id, parent_link_id
        )
        world_to_parent = pybullet.invertTransform(
            parent_to_world[0], parent_to_world[1]
        )
        self_to_parent = pybullet.multiplyTransforms(
            world_to_parent[0],
            world_to_parent[1],
            self_to_world[0],
            self_to_world[1],
        )
    return self_to_parent


def step_and_sleep(seconds=np.inf):
    for i in itertools.count():
        pybullet.stepSimulation()
        time.sleep(pybullet_planning.get_time_step())
        if int(round(i * pybullet_planning.get_time_step())) >= seconds:
            break


def get_camera_image(
    T_cam2world,
    fovy,
    height,
    width,
    far=1000,
    near=0.01,
):
    # T_cam2world -> view_matrix
    view_matrix = T_cam2world.copy()
    view_matrix[:3, 3] = 0
    view_matrix[3, :3] = np.linalg.inv(T_cam2world)[:3, 3]
    view_matrix[:, 1] *= -1
    view_matrix[:, 2] *= -1
    view_matrix = view_matrix.flatten()

    projection_matrix = pybullet.computeProjectionMatrixFOV(
        fov=np.rad2deg(fovy),
        aspect=1.0 * width / height,
        farVal=far,
        nearVal=near,
    )
    _, _, rgba, depth, segm = pybullet.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
    )
    rgb = rgba[:, :, :3]
    depth = np.asarray(depth, dtype=np.float32).reshape(height, width)
    depth = far * near / (far - (far - near) * depth)
    depth[segm == -1] = np.nan
    return rgb, depth, segm


def draw_camera(
    fovy,
    height,
    width,
    pose=None,
    marker_height=0.1,
    marker_color=(0, 0.9, 0.9),
    marker_width=2,
    **kwargs,
):
    aspect_ratio = width / height
    fovx = 2 * np.arctan(np.tan(fovy * 0.5) * aspect_ratio)

    x = marker_height * np.tan(fovx * 0.5)
    y = marker_height * np.tan(fovy * 0.5)
    z = marker_height

    # combine the points into the vertices of an FOV visualization
    points = np.array(
        [(0, 0, 0), (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)],
        dtype=float,
    )

    # create line segments for the FOV visualization
    # a segment from the origin to each bound of the FOV
    segments = np.column_stack((np.zeros_like(points), points)).reshape(
        (-1, 3)
    )

    # add a loop for the outside of the FOV then reshape
    # the whole thing into multiple line segments
    segments = np.vstack((segments, points[[1, 2, 2, 3, 3, 4, 4, 1]])).reshape(
        (-1, 2, 3)
    )

    if pose is not None:
        segments = segments.reshape(-1, 3)
        segments = geometry.transform_points(
            segments, geometry.transformation_matrix(*pose)
        )
        segments = segments.reshape(-1, 2, 3)

    lines = []
    for segment in segments:
        lines.append(
            pybullet_planning.add_line(
                segment[0],
                segment[1],
                color=marker_color,
                width=marker_width,
                **kwargs,
            )
        )
    return lines


def duplicate(body_id, visual=True, collision=True, **kwargs):
    if visual:
        visual_data = pybullet.getVisualShapeData(body_id)
        assert len(visual_data) == 1
        visual_file = visual_data[0][4].decode()
    else:
        visual_file = None

    if collision:
        collision_file = pybullet.getUserData(
            pybullet.getUserDataId(body_id, "collision_file")
        ).decode()
    else:
        collision_file = None

    return create_mesh_body(
        visual_file=visual_file,
        collision_file=collision_file,
        **kwargs,
    )


@contextlib.contextmanager
def stash_objects(object_ids):
    try:
        with pybullet_planning.LockRenderer(), pybullet_planning.WorldSaver():
            for obj in object_ids:
                pybullet_planning.set_pose(obj, ((0, 0, 1000), (0, 0, 0, 1)))
            yield
    finally:
        pass
