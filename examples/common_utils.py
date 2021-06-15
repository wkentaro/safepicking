import shlex
import subprocess

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


def create_pile(class_ids, num_instances, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()

    x = (-0.2, 0.2)
    y = (-0.2, 0.2)
    z = 0.5

    bin_unique_id = mercury.pybullet.create_bin(
        X=x[1] - x[0],
        Y=y[1] - y[0],
        Z=z / 2,
    )

    unique_ids = []
    class_ids = random_state.choice(class_ids, num_instances).tolist()
    while class_ids:
        class_id = class_ids.pop()

        position = random_state.uniform([x[0], y[0], z], [x[1], y[1], z])
        quaternion = random_state.random((4,))
        quaternion /= np.linalg.norm(quaternion)

        coord = mercury.geometry.Coordinate(position, quaternion=quaternion)

        visual_file = mercury.datasets.ycb.get_visual_file(class_id=class_id)
        collision_file = mercury.pybullet.get_collision_file(visual_file)
        unique_id = mercury.pybullet.create_mesh_body(
            visual_file=visual_file,
            collision_file=collision_file,
            mass=mercury.datasets.ycb.masses[class_id],
            position=coord.position,
            quaternion=coord.quaternion,
        )
        p.addUserData(unique_id, "class_id", str(class_id))

        for _ in range(1000):
            p.stepSimulation()
            if np.linalg.norm(p.getBaseVelocity(unique_id)[0]) < 1e-12:
                break

        aabb_min, aabb_max = p.getAABB(bin_unique_id)

        position, _ = p.getBasePositionAndOrientation(unique_id)
        if not (
            (aabb_min[0] < position[0] < aabb_max[0])
            and (aabb_min[1] < position[1] < aabb_max[1])
        ):
            p.removeBody(unique_id)
            class_ids.append(class_id)
        else:
            unique_ids.append(unique_id)

    for _ in range(250):
        coord = mercury.geometry.Coordinate(*pp.get_pose(bin_unique_id))
        coord.translate([0, 0, -0.001], wrt="world")
        pp.set_pose(bin_unique_id, coord.pose)
        for _ in range(100):
            p.stepSimulation()
            if all(
                np.linalg.norm(p.getBaseVelocity(unique_id)[0]) < 1e-12
                for unique_id in mercury.pybullet.get_body_unique_ids()
            ):
                break
    p.removeBody(bin_unique_id)

    return unique_ids


def load_pile(base_pose, npz_file, mass=None):
    data = np.load(npz_file)
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


def git_hash(cwd=None, log_dir=None):
    cmd = "git diff HEAD --"
    diff = subprocess.check_output(shlex.split(cmd), cwd=cwd).decode()
    if diff:
        if log_dir is None:
            raise RuntimeError(
                "There're changes in git, please commit them first"
            )
        else:
            with open(log_dir / "git.diff", "w") as f:
                f.write(diff)
    cmd = "git log --pretty=format:'%h' -n 1"
    return subprocess.check_output(shlex.split(cmd), cwd=cwd).decode().strip()


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
