#!/usr/bin/env python

import collections
import itertools
import time

import gym
from loguru import logger
import numpy as np
import path
import pybullet as p
import pybullet_planning as pp

from yarr.agents.agent import ActResult
from yarr.envs.env import Env
from yarr.utils.observation_type import ObservationElement
from yarr.utils.transition import Transition

import mercury

import common_utils


home = path.Path("~").expanduser()


class PickFromPileEnv(Env):

    PILES_DIR = home / "data/mercury/pile_generation"
    PILE_CENTER = (0.5, 0, 0)

    CLASS_IDS = [2, 3, 5, 11, 12, 15, 16]

    Z_TARGET = 0.2

    DP = 0.05
    DR = np.deg2rad(22.5)

    def __init__(
        self,
        gui=True,
        mp4=None,
        reward_time=0,
        use_reward_translation=False,
        use_reward_dz=False,
        use_reward_max_velocity=False,
    ):
        super().__init__()

        self._gui = gui
        self._mp4 = mp4
        self._reward_time = reward_time
        self._use_reward_translation = use_reward_translation
        self._use_reward_dz = use_reward_dz
        self._use_reward_max_velocity = use_reward_max_velocity

        self.plane = None
        self.ri = None

        dxs = [-self.DP, 0, self.DP]
        dys = [-self.DP, 0, self.DP]
        dzs = [-self.DP, 0, self.DP]
        das = [-self.DR, 0, self.DR]
        dbs = [-self.DR, 0, self.DR]
        dgs = [-self.DR, 0, self.DR]
        self.actions = list(itertools.product(dxs, dys, dzs, das, dbs, dgs))
        self.actions.remove((0, 0, 0, 0, 0, 0))

        grasp_flags = gym.spaces.Box(low=0, high=1, shape=(8,), dtype=np.uint8)
        object_labels = gym.spaces.Box(
            low=0,
            high=1,
            shape=(8, len(self.CLASS_IDS)),
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
                grasp_flags=grasp_flags,
                object_labels=object_labels,
                object_poses=object_poses,
            )
        )

    @property
    def episode_length(self):
        return 10

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
                i = random_state.randint(0, 1000)
            pile_file = self.PILES_DIR / f"{i:08d}.npz"

        if not pp.is_connected():
            pp.connect(use_gui=self._gui, mp4=self._mp4)
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
            suction_max_force=None,
            suction_surface_threshold=np.deg2rad(20),
            suction_surface_alignment=False,
            planner="RRTConnect",
        )
        c_cam_to_ee = mercury.geometry.Coordinate()
        c_cam_to_ee.translate([0, -0.05, -0.1])
        self.ri.add_camera(
            pose=c_cam_to_ee.pose,
            height=240,
            width=320,
        )

        data = dict(np.load(pile_file))

        is_partially_occluded = (0.2 < data["visibility"]) & (
            data["visibility"] < 0.9
        )
        if is_partially_occluded.sum() == 0:
            if raise_on_failure:
                raise RuntimeError("no partially occluded object is found")
            else:
                return self.reset()
        else:
            target_index = random_state.choice(
                np.where(is_partially_occluded)[0]
            )

        num_instances = len(data["class_id"])
        object_ids = []
        for i in range(num_instances):
            class_id = data["class_id"][i]
            position = data["position"][i]
            quaternion = data["quaternion"][i]

            position += self.PILE_CENTER

            visual_file = mercury.datasets.ycb.get_visual_file(
                class_id=class_id
            )
            collision_file = mercury.pybullet.get_collision_file(visual_file)

            with pp.LockRenderer():
                object_id = mercury.pybullet.create_mesh_body(
                    # visual_file=visual_file,
                    visual_file=collision_file,
                    collision_file=collision_file,
                    mass=mercury.datasets.ycb.masses[class_id],
                    position=position,
                    quaternion=quaternion,
                )

            if i == target_index:
                mercury.pybullet.duplicate(
                    object_id,
                    collision=False,
                    texture=False,
                    rgba_color=(0, 1, 0, 0.5),
                    position=position,
                    quaternion=quaternion,
                    mesh_scale=(1.05, 1.05, 1.05),
                )

            object_ids.append(object_id)

        for object_id in object_ids:
            if mercury.pybullet.is_colliding(object_id, ids2=[self.ri.robot]):
                if raise_on_failure:
                    raise RuntimeError("object is colliding with robot")
                else:
                    return self.reset()

        self.object_ids = object_ids
        self.target_object_id = object_ids[target_index]

        self._i = 0
        self._z_min_init = pp.get_aabb(self.target_object_id)[0][2]
        self._z_min_prev = self._z_min_init
        self.translations = collections.defaultdict(float)
        self.max_velocities = collections.defaultdict(float)

        return self.get_obs()

    def get_object_state(self, object_ids, target_object_id):
        grasp_flags = np.zeros((len(object_ids),), dtype=np.uint8)
        object_labels = np.zeros(
            (len(object_ids), len(self.CLASS_IDS)), dtype=np.uint8
        )
        object_poses = np.zeros((len(object_ids), 7), dtype=np.float32)
        for i, object_id in enumerate(object_ids):
            grasp_flags[i] = object_id == target_object_id
            object_to_world = pp.get_pose(object_id)
            class_id = common_utils.get_class_id(object_id)
            object_label = self.CLASS_IDS.index(class_id)
            object_labels[i] = np.eye(len(self.CLASS_IDS))[object_label]
            object_poses[i] = np.hstack(object_to_world)
        return grasp_flags, object_labels, object_poses

    def get_obs(self):
        grasp_flags, object_labels, object_poses = self.get_object_state(
            self.object_ids, self.target_object_id
        )
        obs = dict(
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

    def get_demo_action(self):
        pose = mercury.pybullet.get_pose(self.target_object_id)

        min_distances = {}
        for action_index, action in enumerate(self.actions):
            dx, dy, dz, da, db, dg = action

            c = mercury.geometry.Coordinate(*pose)
            c.translate([dx, dy, dz], wrt="world")
            c.rotate([da, db, dg], wrt="world")

            MAX_DISTANCE = 0.1

            min_distance = MAX_DISTANCE
            with pp.LockRenderer(), pp.WorldSaver():
                pp.set_pose(self.target_object_id, c.pose)

                z_min = pp.get_aabb(self.target_object_id)[0][2]
                if not (z_min >= (self._z_min_init - 0.01)):
                    continue

                for object_id in [self.plane] + self.object_ids:
                    if object_id == self.target_object_id:
                        continue
                    for point in pp.body_collision_info(
                        self.target_object_id,
                        object_id,
                        max_distance=MAX_DISTANCE,
                    ):
                        min_distance = min(min_distance, point[8])
            min_distances[action_index] = min_distance

        action_pz = self.actions.index((0, 0, self.DP, 0, 0, 0))
        if min_distances[action_pz] >= 0:
            action = action_pz
        else:
            i = np.array(list(min_distances.keys()))
            p = np.array(list(min_distances.values()))
            p = (p - p.min()) / (p.max() - p.min())
            p = p / p.sum()
            action = np.random.choice(i, p=p)

        return ActResult(action=action)

    def validate_action(self, act_result):
        dx, dy, dz, da, db, dg = self.actions[act_result.action]

        c = mercury.geometry.Coordinate(
            *mercury.pybullet.get_pose(self.target_object_id)
        )
        c.translate([dx, dy, dz], wrt="world")
        c.rotate([da, db, dg], wrt="world")

        with pp.LockRenderer(), pp.WorldSaver():
            pp.set_pose(self.target_object_id, c.pose)
            z_min = pp.get_aabb(self.target_object_id)[0][2]
            return z_min >= (self._z_min_init - 0.01)

    def step(self, act_result):
        if not self.validate_action(act_result):
            raise RuntimeError

        dx, dy, dz, da, db, dg = action = self.actions[act_result.action]

        with np.printoptions(precision=2):
            logger.info(f"[{act_result.action}] {np.array(action)}")

        pose1 = mercury.pybullet.get_pose(self.target_object_id)

        c = mercury.geometry.Coordinate(*pose1)
        c.translate([dx, dy, dz], wrt="world")
        c.rotate([da, db, dg], wrt="world")

        pose2 = c.pose

        poses = {}
        for object_id in self.object_ids:
            if object_id == self.target_object_id:
                continue
            poses[object_id] = pp.get_pose(object_id)
        translations = collections.defaultdict(float)
        max_velocities = collections.defaultdict(float)

        def step_callback():
            for object_id in self.object_ids:
                if object_id == self.target_object_id:
                    continue

                pose = pp.get_pose(object_id)

                translations[object_id] += np.linalg.norm(
                    np.array(poses[object_id][0]) - np.array(pose[0])
                )

                poses[object_id] = pose

                max_velocities[object_id] = max(
                    max_velocities[object_id],
                    np.linalg.norm(pp.get_velocity(object_id)[0]),
                )

        for pose in pp.interpolate_poses(
            pose1, pose2, pos_step_size=0.001, ori_step_size=np.pi / 180
        ):
            pp.set_pose(self.target_object_id, pose)
            pp.step_simulation()
            pp.set_pose(self.target_object_id, pose)
            step_callback()
            if self._gui:
                time.sleep(pp.get_time_step())

        for _ in range(int(round(1 / pp.get_time_step()))):
            pp.set_pose(self.target_object_id, pose)
            pp.step_simulation()
            pp.set_pose(self.target_object_id, pose)
            step_callback()
            if self._gui:
                time.sleep(pp.get_time_step())

        mercury.pybullet.duplicate(
            self.target_object_id,
            collision=False,
            texture=False,
            rgba_color=(0, 1, 0, 0.5),
            position=pose[0],
            quaternion=pose[1],
            mesh_scale=(1.05, 1.05, 1.05),
        )

        # ---------------------------------------------------------------------

        self._i += 1

        for object_id in self.object_ids:
            if object_id == self.target_object_id:
                continue
            self.translations[object_id] = (
                self.translations[object_id] + translations[object_id]
            )
            self.max_velocities[object_id] = max(
                self.max_velocities[object_id], max_velocities[object_id]
            )

        z_min = pp.get_aabb(self.target_object_id)[0][2]

        # primary task
        if z_min >= self.Z_TARGET:
            terminal = True
            reward = 1
        elif self._i == self.episode_length:
            terminal = True
            reward = 0
        else:
            terminal = False
            reward = 0

        # reward shaping
        if not self.eval:
            reward += self._reward_time

            if self._use_reward_translation:
                reward += -sum(translations.values())

            if self._use_reward_dz:
                reward += (z_min - self._z_min_prev) / (
                    self.Z_TARGET - self._z_min_init
                )

            if self._use_reward_max_velocity:
                reward += -sum(max_velocities.values())

        self._z_min_prev = z_min

        logger.info(f"Reward={reward:.2f}, Terminal={terminal}")

        info = {"translation": sum(translations.values())}
        if terminal:
            info["max_velocity"] = max(self.max_velocities.values())

        return Transition(
            observation=self.get_obs(),
            reward=reward,
            terminal=terminal,
            info=info,
        )
