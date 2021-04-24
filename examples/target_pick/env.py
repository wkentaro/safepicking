#!/usr/bin/env python

import itertools
import time

import gym
from loguru import logger
import numpy as np
import path
import pybullet as p
import pybullet_planning as pp

from yarr.envs.env import Env
from yarr.utils.observation_type import ObservationElement
from yarr.utils.transition import Transition

import mercury

import utils


here = path.Path(__file__).abspath().parent


class PickFromPileEnv(Env):

    piles_dir = here / "logs/export_pile"
    class_ids = [2, 3, 5, 11, 12, 15, 16]

    def __init__(self, gui=True, retime=1, planner="RRTConnect"):
        self.gui = gui
        self.retime = retime
        self.planner = planner

        self.plane = None
        self.ri = None
        self.pile_files = list(sorted(self.piles_dir.listdir()))

        dpos = 0.05
        drot = np.deg2rad(15)
        actions = []
        dz = dpos / 2
        for dx in [0, dpos, -dpos]:
            for dy in [0, dpos, -dpos]:
                for da in [0, drot, -drot]:
                    for db in [0, drot, -drot]:
                        for dg in [0, drot, -drot]:
                            actions.append([dx, dy, dz, da, db, dg])
        self.actions = actions

        grasp_position = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32,
        )
        past_actions = gym.spaces.Box(
            low=0,
            high=1,
            shape=(4, len(self.actions)),
            dtype=np.uint8,
        )
        grasp_flags = gym.spaces.Box(low=0, high=1, shape=(8,), dtype=np.uint8)
        object_labels = gym.spaces.Box(
            low=0,
            high=1,
            shape=(8, len(self.class_ids)),
            dtype=np.uint8,
        )
        object_poses = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8, 7),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            dict(
                grasp_position=grasp_position,
                past_actions=past_actions,
                grasp_flags=grasp_flags,
                object_labels=object_labels,
                object_poses=object_poses,
            )
        )

    @property
    def action_shape(self):
        return ()

    def env(self):
        return

    @property
    def observation_elements(self):
        elements = []
        for name, space in self.observation_space.spaces.items():
            elements.append(ObservationElement(name, space.shape, space.dtype))
        return elements

    # -------------------------------------------------------------------------

    def shutdown(self):
        pp.disconnect()

    def launch(self):
        pass

    def reset(self, random_state=None, pile_file=None):
        if random_state is None:
            random_state = np.random.RandomState()
        if pile_file is None:
            pile_file = random_state.choice(self.pile_files)

        if not pp.is_connected():
            pp.connect(use_gui=self.gui)
            pp.add_data_path()

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=90,
            cameraPitch=-60,
            cameraTargetPosition=(0, 0, 0),
        )
        self.plane = p.loadURDF("plane.urdf")

        self.ri = mercury.pybullet.PandaRobotInterface(
            suction_max_force=10, planner=self.planner
        )
        c_cam_to_ee = mercury.geometry.Coordinate()
        c_cam_to_ee.translate([0, -0.05, -0.1])
        self.ri.add_camera(
            pose=c_cam_to_ee.pose,
            height=240,
            width=320,
        )

        data = dict(np.load(pile_file))

        is_occluded = data["visibility"] < 0.9
        if is_occluded.sum() == 0:
            raise RuntimeError("no occluded object is found")
        else:
            target_index = random_state.choice(np.where(is_occluded)[0])

        pile_center = [0.5, 0, 0]

        num_instances = len(data["class_id"])
        object_ids = []
        for i in range(num_instances):
            class_id = data["class_id"][i]
            position = data["position"][i]
            quaternion = data["quaternion"][i]

            position += pile_center

            visual_file = mercury.datasets.ycb.get_visual_file(
                class_id=class_id
            )
            collision_file = mercury.pybullet.get_collision_file(visual_file)

            if i == target_index:
                rgba_color = (1, 0, 0)
            else:
                rgba_color = (0.5, 0.5, 0.5)

            class_name = mercury.datasets.ycb.class_names[class_id]
            visibility = data["visibility"][i]
            logger.info(
                f"class_id={class_id:02d}, "
                f"class_name={class_name}, "
                f"visibility={visibility:.1%}"
            )

            with pp.LockRenderer():
                object_id = mercury.pybullet.create_mesh_body(
                    visual_file=visual_file,
                    collision_file=collision_file,
                    mass=mercury.datasets.ycb.masses[class_id],
                    position=position,
                    quaternion=quaternion,
                    rgba_color=rgba_color,
                    texture=False,
                )
            object_ids.append(object_id)

        c = mercury.geometry.Coordinate(*self.ri.get_pose("camera_link"))
        c.position = pp.get_pose(object_ids[target_index])[0]
        c.position[2] += 0.5
        for i in itertools.count():
            if i == 10:
                return self.reset()

            j = self.ri.solve_ik(
                c.pose, move_target=self.ri.robot_model.camera_link
            )
            js = self.ri.planj(j)
            for j in js:
                for _ in self.ri.movej(j):
                    p.stepSimulation()
                    if self.gui:
                        time.sleep(1 / 240 / self.retime)

            rgb, depth, segm = self.ri.get_camera_image()

            for _ in self.ri.random_grasp(
                depth,
                segm,
                bg_object_ids=[self.plane],
                object_ids=object_ids,
                target_object_ids=[object_ids[target_index]],
                random_state=random_state,
                noise=False,
            ):
                p.stepSimulation()
                if self.gui:
                    time.sleep(1 / 240 / self.retime)

            for _ in range(240):
                p.stepSimulation()
                self.ri.step_simulation()
                if self.gui:
                    time.sleep(1 / 240 / self.retime)

            if (
                self.ri.gripper.check_grasp()
                and self.ri.attachments[0].child == object_ids[target_index]
            ):
                break
            else:
                self.ri.ungrasp()

        self.grasp_position = self.ri.get_pose("tipLink")[0].astype(np.float32)
        self.object_ids = object_ids
        self.target_object_id = object_ids[target_index]
        self.past_actions = np.zeros((4, len(self.actions)), dtype=np.uint8)

        self.i = 0

        return self.get_obs()

    def get_obs(self):
        ri = self.ri
        object_ids = self.object_ids
        target_object_id = self.target_object_id

        grasp_flags = np.zeros((len(object_ids),), dtype=np.uint8)
        object_labels = np.zeros(
            (len(object_ids), len(self.class_ids)), dtype=np.uint8
        )
        object_poses = np.zeros((len(object_ids), 7), dtype=np.float32)
        for i, object_id in enumerate(object_ids):
            if object_id == target_object_id:
                grasp_flags[i] = 1
                object_to_ee = ri.attachments[0].grasp_pose
                ee_to_world = ri.get_pose("tipLink")
                object_to_world = pp.multiply(ee_to_world, object_to_ee)
            else:
                grasp_flags[i] = 0
                object_to_world = pp.get_pose(object_id)
            class_id = utils.get_class_id(object_id)
            object_label = self.class_ids.index(class_id)
            object_labels[i] = np.eye(len(self.class_ids))[object_label]
            object_poses[i] = np.hstack(object_to_world)

        obs = dict(
            grasp_position=self.grasp_position,
            past_actions=self.past_actions,
            grasp_flags=grasp_flags,
            object_labels=object_labels,
            object_poses=object_poses,
        )

        for key, space in self.observation_space.spaces.items():
            assert obs[key].shape == space.shape, (
                key,
                obs[key].shape,
                space.shape,
            )
            assert obs[key].dtype == space.dtype, (
                key,
                obs[key].dtype,
                space.dtype,
            )

        return obs

    def validate_action(self, act_result):
        action = self.actions[act_result.action]
        dx, dy, dz, da, db, dg = action

        c = mercury.geometry.Coordinate(*self.ri.get_pose("tipLink"))
        c.translate([dx, dy, dz], wrt="world")
        c.rotate([da, db, dg], wrt="world")

        return self.ri.solve_ik(c.pose)

    def step(self, act_result):
        if hasattr(act_result, "j"):
            j = act_result.j
        else:
            j = self.validate_action(act_result)

        if j is None:
            logger.error("failed to solve IK")
            return Transition(
                observation=self.get_obs(), reward=0, terminal=True
            )

        for _ in self.ri.movej(j, speed=0.001):
            p.stepSimulation()
            self.ri.step_simulation()
            if self.gui:
                time.sleep(1 / 240 / self.retime)

        self.i += 1

        if self.i == 5:
            for _ in self.ri.movej(self.ri.homej, speed=0.001, timeout=30):
                p.stepSimulation()
                self.ri.step_simulation()
                if self.gui:
                    time.sleep(1 / 240 / self.retime)
            reward = int(self.ri.gripper.check_grasp())
            terminal = True
        else:
            reward = 0
            terminal = False

        self.past_actions = np.r_[
            self.past_actions[1:],
            np.eye(len(self.actions), dtype=np.uint8)[act_result.action][None],
        ]

        return Transition(
            observation=self.get_obs(), reward=reward, terminal=terminal
        )
