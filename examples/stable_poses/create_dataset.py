#!/usr/bin/env python

import argparse
import pickle
import time
import tqdm

import numpy as np
import pybullet_planning as pp
import sklearn.neighbors

import mercury

from pick_and_place_env import PickAndPlaceEnv


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--scene-index", type=int, required=True)
parser.add_argument("--gui", action="store_true", help="gui")
args = parser.parse_args()

env = PickAndPlaceEnv(class_ids=[5], gui=args.gui)
env.random_state = np.random.RandomState(1)

env.reset()

pose_init = pp.get_pose(env.fg_object_id)
aabb_center = env.PILE_POSITION
aabb_min = aabb_center - 0.3
aabb_max = aabb_center + 0.3
pp.draw_aabb((aabb_min, aabb_max))

pp.set_pose(env.fg_object_id, ([1, 1, 1], [0, 0, 0, 1]))
for _ in range(2400):
    pp.step_simulation()
for object_id in env.object_ids:
    if object_id != env.fg_object_id:
        pp.set_mass(object_id, 0)

rgb, depth, segm = env.ri.get_camera_image()
K = env.ri.get_opengl_intrinsic_matrix()
pcd = mercury.geometry.pointcloud_from_depth(
    depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
)
cam_pose = env.ri.get_pose("camera_link")

XY = np.random.uniform(aabb_min[0:2], aabb_max[0:2], size=(100, 2))
A = np.linspace(-np.pi, np.pi, num=5, endpoint=False)
B = np.linspace(-np.pi, np.pi, num=5, endpoint=False)
G = np.linspace(-np.pi, np.pi, num=5, endpoint=False)

pcd_cam = mercury.geometry.pointcloud_from_depth(
    depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
)
pcd_world = mercury.geometry.transform_points(
    pcd_cam, mercury.geometry.transformation_matrix(*cam_pose)
)

pcd_file = mercury.datasets.ycb.get_pcd_file(class_id=5)
pcd_object = np.loadtxt(pcd_file)

kdtree = sklearn.neighbors.KDTree(pcd_world.reshape(-1, 3)[:, :2])
width, height = rgb.shape[:2]

data = dict(
    rgb=rgb,
    depth=depth,
    segm=segm,
    K=K,
    yx=[],
    z=[],
    euler=[],
    auc=[],
)
for x, y in tqdm.tqdm(XY):
    a = np.random.choice(A)
    b = np.random.choice(B)
    g = np.random.choice(G)
    with pp.WorldSaver():
        c = mercury.geometry.Coordinate(position=[x, y, 0])
        c.quaternion = mercury.geometry.quaternion_from_euler([a, b, g])
        pp.set_pose(env.fg_object_id, c.pose)

        while mercury.pybullet.is_colliding(env.fg_object_id):
            c.translate([0, 0, 0.001], wrt="world")
            pp.set_pose(env.fg_object_id, c.pose)

        pose1 = pp.get_pose(env.fg_object_id)
        # debug = pp.add_text("stabilizing", position=c.pose[0] + [0, 0, 0.1])  # NOQA
        for i in range(2400):
            if i % 24 == 0:
                pp.set_velocity(
                    env.fg_object_id,
                    np.random.uniform(-0.01, 0.01, (3,)),
                    np.random.uniform(-0.1, 0.1, (3,)),
                )
            pp.step_simulation()
            if args.gui:
                time.sleep(pp.get_time_step() / 10)
        # pp.remove_debug(debug)

        # debug = pp.add_text("validating", position=c.pose[0] + [0, 0, 0.1])  # NOQA
        # for _ in range(240):
        #     pp.step_simulation()
        # pp.remove_debug(debug)

        pose2 = pp.get_pose(env.fg_object_id)

        # Save

        distance, index = kdtree.query([pose2[0][:2]])
        distance = distance[0, 0]
        index = index[0, 0]
        if distance < 0.01:
            y = index // width
            x = index % width
            data["yx"].append((y, x))
            data["z"].append(pose2[0][2])
            data["euler"].append(mercury.geometry.Coordinate(*pose2).euler)
            data["auc"].append(1)

        pcd2 = mercury.geometry.transform_points(
            pcd_object, mercury.geometry.transformation_matrix(*pose2)
        )
        pcd1 = mercury.geometry.transform_points(
            pcd_object, mercury.geometry.transformation_matrix(*pose1)
        )
        auc = mercury.geometry.average_distance_auc(
            pcd1, pcd2, max_threshold=0.1
        )

        distance, index = kdtree.query([pose1[0][:2]])
        distance = distance[0, 0]
        index = index[0, 0]
        if distance < 0.01:
            y = index // width
            x = index % width
            data["yx"].append((y, x))
            data["z"].append(pose1[0][2])
            data["euler"].append(mercury.geometry.Coordinate(*pose1).euler)
            data["auc"].append(auc)
data["yx"] = np.array(data["yx"])
data["z"] = np.array(data["z"])
data["euler"] = np.array(data["euler"])
data["auc"] = np.array(data["auc"])
with open(f"{args.scene_index:08d}.pkl", "wb") as f:
    pickle.dump(data, f)
