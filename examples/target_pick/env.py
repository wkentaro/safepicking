#!/usr/bin/env python

import copy
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

import common_utils


home = path.Path("~").expanduser()


class PickFromPileEnv(Env):

    piles_dir = home / "data/mercury/pile_generation"
    class_ids = [2, 3, 5, 11, 12, 15, 16]

    def __init__(
        self,
        gui=True,
        retime=1,
        planner="RRTConnect",
        pose_noise=False,
        easy=False,
        suction_max_force=None,
        reward="max_velocities",
        action_discretization=3,
    ):
        super().__init__()

        self._gui = gui
        self._retime = retime
        self.planner = planner
        self._pose_noise = pose_noise
        self._easy = easy
        self._suction_max_force = suction_max_force

        if reward == "max_velocities":
            assert suction_max_force is None
        else:
            assert reward in ["completion", "completion_shaped"]
        self._reward = reward

        self.plane = None
        self.ri = None

        dpos = 0.4 / 8
        drot = np.deg2rad(120 / 8)
        num = action_discretization
        dx_options = np.linspace(-dpos, dpos, num=num)
        dy_options = np.linspace(-dpos, dpos, num=num)
        dz_options = np.linspace(-dpos, dpos, num=num)
        da_options = np.linspace(-drot, drot, num=num)
        db_options = np.linspace(-drot, drot, num=num)
        dg_options = np.linspace(-drot, drot, num=num)

        actions = list(
            itertools.product(
                dx_options,
                dy_options,
                dz_options,
                da_options,
                db_options,
                dg_options,
            )
        )
        self.actions = actions

        grasp_pose = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32,
        )
        ee_pose = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32,
        )
        past_actions = gym.spaces.Box(
            low=0,
            high=1,
            shape=(4, len(self.actions)),
            dtype=np.uint8,
        )
        past_grasped_object_poses = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4, 7),
            dtype=np.float32,
        )
        grasp_flags_openloop = gym.spaces.Box(
            low=0, high=1, shape=(8,), dtype=np.uint8
        )
        object_labels_openloop = gym.spaces.Box(
            low=0,
            high=1,
            shape=(8, len(self.class_ids)),
            dtype=np.uint8,
        )
        object_poses_openloop = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8, 7),
            dtype=np.float32,
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
                grasp_pose=grasp_pose,
                ee_pose=ee_pose,
                past_actions=past_actions,
                past_grasped_object_poses=past_grasped_object_poses,
                grasp_flags_openloop=grasp_flags_openloop,
                object_labels_openloop=object_labels_openloop,
                object_poses_openloop=object_poses_openloop,
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
        raise_on_failure = pile_file is not None

        if random_state is None:
            random_state = np.random.RandomState()
        if pile_file is None:
            if self.eval:
                i = random_state.randint(1000, 1200)
            else:
                if self._easy:
                    i = random_state.randint(1000, 1200)
                else:
                    i = random_state.randint(0, 1000)
            pile_file = self.piles_dir / f"{i:08d}.npz"

        if not pp.is_connected():
            pp.connect(use_gui=self._gui)
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
            suction_max_force=self._suction_max_force, planner="RRTConnect"
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
            if raise_on_failure:
                raise RuntimeError("no occluded object is found")
            else:
                return self.reset()
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
                if raise_on_failure:
                    raise RuntimeError("random grasping failed")
                else:
                    return self.reset()

            j = self.ri.solve_ik(
                c.pose, move_target=self.ri.robot_model.camera_link
            )
            if j is None:
                continue

            self.ri.setj(j)

            rgb, depth, segm = self.ri.get_camera_image()
            object_state = self.get_object_state(
                object_ids=object_ids,
                target_object_id=object_ids[target_index],
                random_state=copy.deepcopy(random_state),
            )

            try:
                for _ in self.ri.random_grasp(
                    depth,
                    segm,
                    mask=segm == object_ids[target_index],
                    bg_object_ids=[self.plane],
                    object_ids=object_ids,
                    random_state=random_state,
                    noise=False,
                ):
                    p.stepSimulation()
                    if self._gui:
                        time.sleep(pp.get_time_step() / self._retime)
            except RuntimeError:
                if raise_on_failure:
                    raise RuntimeError("random grasping failed")
                else:
                    return self.reset()

            for _ in range(int(round(1 / pp.get_time_step()))):
                p.stepSimulation()
                if self._suction_max_force is not None:
                    self.ri.step_simulation()
                if self._gui:
                    time.sleep(pp.get_time_step() / self._retime)

            if (
                self.ri.gripper.check_grasp()
                and self.ri.attachments[0].child == object_ids[target_index]
            ):
                break
            else:
                self.ri.ungrasp()
        self.ri.planner = self.planner  # Use specified planner in step()

        self.grasp_pose = np.hstack(self.ri.get_pose("tipLink")).astype(
            np.float32
        )
        self.object_state = object_state
        self.object_ids = object_ids
        self.target_object_id = object_ids[target_index]
        self.past_actions = np.zeros((4, len(self.actions)), dtype=np.uint8)
        self.past_grasped_object_poses = np.zeros((4, 7), dtype=np.float32)

        self.i = 0

        return self.get_obs()

    def get_object_state(
        self, object_ids, target_object_id, random_state=None
    ):
        if random_state is None:
            random_state = np.random.RandomState()
        grasp_flags = np.zeros((len(object_ids),), dtype=np.uint8)
        object_labels = np.zeros(
            (len(object_ids), len(self.class_ids)), dtype=np.uint8
        )
        object_poses = np.zeros((len(object_ids), 7), dtype=np.float32)
        for i, object_id in enumerate(object_ids):
            if object_id == target_object_id and self.ri.attachments:
                grasp_flags[i] = 1
                object_to_ee = self.ri.attachments[0].grasp_pose
                ee_to_world = self.ri.get_pose("tipLink")
                object_to_world = pp.multiply(ee_to_world, object_to_ee)
            else:
                grasp_flags[i] = object_id == target_object_id
                object_to_world = pp.get_pose(object_id)
                if self._pose_noise:
                    object_to_world = (
                        object_to_world[0] + random_state.normal(0, 0.003, 3),
                        object_to_world[1] + random_state.normal(0, 0.01, 4),
                    )
            class_id = common_utils.get_class_id(object_id)
            object_label = self.class_ids.index(class_id)
            object_labels[i] = np.eye(len(self.class_ids))[object_label]
            object_poses[i] = np.hstack(object_to_world)
        return grasp_flags, object_labels, object_poses

    def get_obs(self):
        ee_pose = np.hstack(self.ri.get_pose("tipLink")).astype(np.float32)
        (
            grasp_flags_openloop,
            object_labels_openloop,
            object_poses_openloop,
        ) = self.object_state
        grasp_flags, object_labels, object_poses = self.get_object_state(
            self.object_ids, self.target_object_id
        )
        assert (grasp_flags == grasp_flags_openloop).all()
        assert (object_labels == object_labels_openloop).all()
        obs = dict(
            grasp_pose=self.grasp_pose,
            grasp_flags_openloop=grasp_flags_openloop,
            object_labels_openloop=object_labels_openloop,
            object_poses_openloop=object_poses_openloop,
            ee_pose=ee_pose,
            past_actions=self.past_actions,
            past_grasped_object_poses=self.past_grasped_object_poses,
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
                observation=self.get_obs(),
                reward=0,
                terminal=True,
                info=dict(max_velocities=None),
            )

        max_velocities = {}

        def step_callback():
            for object_id in self.object_ids:
                if object_id == self.target_object_id:
                    continue
                max_velocities[object_id] = max(
                    max_velocities.get(object_id, 0),
                    np.linalg.norm(pp.get_velocity(object_id)[0]),
                )

        for _ in self.ri.movej(j, speed=0.001):
            p.stepSimulation()
            if self._suction_max_force is not None:
                self.ri.step_simulation()
            if step_callback:
                step_callback()
            if self._gui:
                time.sleep(pp.get_time_step() / self._retime)

        self.i += 1

        if self.i == 5:
            for _ in self.ri.movej(self.ri.homej, speed=0.001, timeout=30):
                p.stepSimulation()
                if self._suction_max_force is not None:
                    self.ri.step_simulation()
                if step_callback:
                    step_callback()
                if self._gui:
                    time.sleep(pp.get_time_step() / self._retime)
            reward = int(self.ri.gripper.check_grasp())
            terminal = True
        else:
            if self.eval:
                reward = 0
            else:
                if self._reward == "completion_shaped":
                    reward = self.i * 0.2 * self.ri.gripper.check_grasp()
                else:
                    assert self._reward in ["completion", "max_velocities"]
            terminal = False

        if self._reward == "max_velocities":
            reward = -sum(max_velocities.values())
        else:
            assert self._reward in ["completion_shaped", "completion"]

        self.past_actions = np.r_[
            self.past_actions[1:],
            np.eye(len(self.actions), dtype=np.uint8)[act_result.action][None],
        ]

        object_to_ee = self.ri.attachments[0].grasp_pose
        ee_to_world = self.ri.get_pose("tipLink")
        object_to_world = pp.multiply(ee_to_world, object_to_ee)
        self.past_grasped_object_poses = np.r_[
            self.past_grasped_object_poses[1:],
            np.hstack(object_to_world).astype(np.float32)[None],
        ]

        return Transition(
            observation=self.get_obs(),
            reward=reward,
            terminal=terminal,
            info=dict(max_velocities=max_velocities),
        )
