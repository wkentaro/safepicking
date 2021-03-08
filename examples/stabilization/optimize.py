#!/usr/bin/env python

import time

import imgviz
import pybullet
import torch

import mercury

from create_pile import get_cad_file
from train import Dataset
from train import get_model


data = Dataset(split="train")
loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)

model = get_model()
model.eval()
model.load_state_dict(
    torch.load("./data/train/20210225_180602/model_best.pth")
)

for i, (poses, _) in enumerate(loader):
    if i == 3:
        break
poses = poses.to("cuda:0")
poses.requires_grad = True

optimizer = torch.optim.Adam([poses], lr=1e-4)

mercury.pybullet.init_world()
pybullet.resetDebugVisualizerCamera(
    cameraDistance=1,
    cameraYaw=-60,
    cameraPitch=-60,
    cameraTargetPosition=(0, 0, 0),
)

class_ids = [2, 15, 11, 3, 2, 12, 11, 15]
unique_ids = []
for i in range(8):
    visual_file = get_cad_file(class_ids[i])
    collision_file = mercury.pybullet.get_collision_file(visual_file)
    unique_id = mercury.pybullet.create_mesh_body(
        visual_file=collision_file,
        collision_file=collision_file,
        position=poses.reshape(8, -1)[i][:3].detach().cpu().numpy(),
        quaternion=poses.reshape(8, -1)[i][3:].detach().cpu().numpy(),
        rgba_color=imgviz.label_colormap()[class_ids[i]] / 255,
    )
    unique_ids.append(unique_id)

for _ in range(1000):
    y_pred = model(poses)

    # loss = y_pred.sum()
    loss = y_pred[0, 5]

    optimizer.zero_grad()
    loss.backward()
    i0 = 7 * 5
    i1 = i0 + 7
    poses.grad[:, :i0] = 0
    poses.grad[:, i1:] = 0
    optimizer.step()

    for i in range(8):
        position = poses.reshape(8, -1)[i][:3].detach().cpu().numpy()
        quaternion = poses.reshape(8, -1)[i][3:].detach().cpu().numpy()
        pybullet.resetBasePositionAndOrientation(
            unique_ids[i], position, quaternion
        )

while True:
    time.sleep(1 / 240)
    pybullet.stepSimulation()
