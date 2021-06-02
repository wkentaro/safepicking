import pickle
import time

import imgviz
import numpy as np
import pybullet_planning as pp
import torch

import mercury

import common_utils
from pick_and_place_env import PickAndPlaceEnv
from train import Unet


model = Unet()
model.load_state_dict(torch.load("model.pth"))

data = pickle.load(
    open("/home/wkentaro/data/mercury/stable_poses/00000001.pkl", "rb")
)
rgb = (data["rgb"] / 255).astype(np.float32).transpose(2, 0, 1)
yx = np.array(data["yx"])
euler = np.array(data["euler"]).astype(np.float32)

A = np.linspace(-np.pi, np.pi, num=5, endpoint=False)
B = np.linspace(-np.pi, np.pi, num=5, endpoint=False)
G = np.linspace(-np.pi, np.pi, num=5, endpoint=False)
if 1:
    a = np.random.choice(A, size=10000)
    b = np.random.choice(B, size=10000)
    g = np.random.choice(G, size=10000)
    euler2 = np.stack([a, b, g], axis=1).astype(np.float32)
    yx2 = np.random.randint(yx.min(axis=0), yx.max(axis=0), size=(10000, 2))

    euler = np.vstack([euler2, euler])
    yx = np.vstack([yx2, yx])
if 0:
    a = np.random.choice(A, size=yx.shape[0])
    b = np.random.choice(B, size=yx.shape[0])
    g = np.random.choice(G, size=yx.shape[0])
    euler = np.stack([a, b, g], axis=1).astype(np.float32)


print(rgb.shape)
print(yx.shape)
print(euler.shape)

if 0:
    with torch.no_grad():
        stable_pred = model(
            torch.as_tensor(rgb[None]),
            torch.as_tensor(yx[None]),
            torch.as_tensor(euler[None]),
        )[0]
else:
    stable_pred = data["auc"]

if 0:
    stable = np.full((240, 240), 0.5)
    for i in range(len(yx)):
        y = yx[i, 0]
        x = yx[i, 1]
        stable[y, x] = stable_pred[i].detach().cpu().numpy()

    stable = imgviz.depth2rgb(stable, min_value=0, max_value=1)
    gray = imgviz.gray2rgb(imgviz.rgb2gray(data["rgb"]))
    stable = np.uint8(gray * 0.3 + stable * 0.7)
    viz = imgviz.tile([data["rgb"], stable])
    viz = imgviz.resize(viz, width=1200)
    imgviz.io.pyglet_imshow(viz)
    imgviz.io.pyglet_run()

env = PickAndPlaceEnv(class_ids=[5], gui=True, mp4="test_data.mp4")
env.random_state = np.random.RandomState(1)
env.reset()

for object_id in env.object_ids:
    if object_id != env.fg_object_id:
        pp.set_mass(object_id, 0)

env.setj_to_camera_pose()
K = env.ri.get_opengl_intrinsic_matrix()
pcd = mercury.geometry.pointcloud_from_depth(
    data["depth"], fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
)
T_cam_to_world = mercury.geometry.transformation_matrix(
    *env.ri.get_pose("camera_link")
)
pcd = mercury.geometry.transform_points(pcd, T_cam_to_world)

pcd_file = mercury.datasets.ycb.get_pcd_file(class_id=5)
pcd_object = np.loadtxt(pcd_file)

for i in np.random.permutation(len(stable_pred)):
    s = stable_pred[i]
    if s < 0.99:
        continue
    y, x = yx[i]
    e = euler[i]

    print(s, (y, x), e)

    position = pcd[y, x]

    with pp.WorldSaver():
        c = mercury.geometry.Coordinate(
            position, mercury.geometry.quaternion_from_euler(e)
        )
        pp.set_pose(env.fg_object_id, c.pose)

        while mercury.pybullet.is_colliding(env.fg_object_id):
            c.translate([0, 0, 0.001], wrt="world")
            pp.set_pose(env.fg_object_id, c.pose)

        common_utils.pause(True)

        pose1 = pp.get_pose(env.fg_object_id)

        for i in range(2400):
            pp.step_simulation()
            time.sleep(pp.get_time_step() / 10)

        pose2 = pp.get_pose(env.fg_object_id)

    pcd2 = mercury.geometry.transform_points(
        pcd_object, mercury.geometry.transformation_matrix(*pose2)
    )
    pcd1 = mercury.geometry.transform_points(
        pcd_object, mercury.geometry.transformation_matrix(*pose1)
    )
    auc = mercury.geometry.average_distance_auc(pcd1, pcd2, max_threshold=0.1)
    print(auc)
