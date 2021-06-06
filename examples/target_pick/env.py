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
        use_reward_translation=False,
        use_reward_max_velocity=False,
        speed=0.01,
        episode_length=10,
    ):
        super().__init__()

        self._gui = gui
        self._mp4 = mp4
        self._use_reward_translation = use_reward_translation
        self._use_reward_max_velocity = use_reward_max_velocity
        self._speed = speed
        self._episode_length = episode_length

        self.plane = None
        self.ri = None

        dxs = [-self.DP, 0, self.DP]
        dys = [-self.DP, 0, self.DP]
        dzs = [-self.DP, 0, self.DP]
        das = [-self.DR, 0, self.DR]
        dbs = [-self.DR, 0, self.DR]
        dgs = [-self.DR, 0, self.DR]
        self.actions = list(itertools.product(dxs, dys, dzs, das, dbs, dgs))

        ee_pose = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32,
        )
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
                ee_pose=ee_pose,
                grasp_flags=grasp_flags,
                object_labels=object_labels,
                object_poses=object_poses,
            )
        )

    @property
    def episode_length(self):
        return self._episode_length

    @property
    def action_shape(self):
        return (2,)

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
                    rgba_color=(0, 1, 0) if i == target_index else (1, 1, 1),
                )

            object_ids.append(object_id)
        target_object_id = object_ids[target_index]

        for object_id in object_ids:
            if mercury.pybullet.is_colliding(object_id, ids2=[self.ri.robot]):
                if raise_on_failure:
                    raise RuntimeError("object is colliding with robot")
                else:
                    return self.reset()

        # grasping

        c = mercury.geometry.Coordinate(*self.ri.get_pose("camera_link"))
        c.position = pp.get_pose(target_object_id)[0]
        c.position[2] += 0.5

        j = self.ri.solve_ik(
            c.pose, move_target=self.ri.robot_model.camera_link
        )
        if j is None:
            if raise_on_failure:
                raise RuntimeError("IK failed to capture object")
            else:
                return self.reset()

        self.ri.setj(j)

        rgb, depth, segm = self.ri.get_camera_image()

        K = self.ri.get_opengl_intrinsic_matrix()

        pcd_in_camera = mercury.geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )

        camera_to_world = self.ri.get_pose("camera_link")
        ee_to_world = self.ri.get_pose("tipLink")
        camera_to_ee = pp.multiply(pp.invert(ee_to_world), camera_to_world)
        pcd_in_ee = mercury.geometry.transform_points(
            pcd_in_camera,
            mercury.geometry.transformation_matrix(*camera_to_ee),
        )

        normals = mercury.geometry.normals_from_pointcloud(pcd_in_ee)

        yx = np.argwhere(segm == target_object_id)

        for y, x in yx[np.random.permutation(len(yx))]:
            position = pcd_in_ee[y, x]
            quaternion = mercury.geometry.quaternion_from_vec2vec(
                [0, 0, 1], normals[y, x]
            )
            ee_to_ee_af = (position, quaternion)
            ee_af_to_world = pp.multiply(ee_to_world, ee_to_ee_af)

            j = self.ri.solve_ik(ee_af_to_world, rotation_axis="z")
            if j is None:
                continue

            obstacles = [self.plane] + object_ids
            obstacles.remove(target_object_id)
            if not self.ri.validatej(j, obstacles=obstacles):
                continue

            self.ri.setj(j)
            break
        else:
            if raise_on_failure:
                raise RuntimeError("Unable to grasp the target object")
            else:
                return self.reset()

        for _ in self.ri.movej(j):
            pp.step_simulation()
            time.sleep(pp.get_time_step())

        try:
            for _ in self.ri.grasp(rotation_axis=True):
                pp.step_simulation()
                time.sleep(pp.get_time_step())
        except RuntimeError:
            if raise_on_failure:
                raise
            else:
                return self.reset()

        obj_to_world = pp.get_pose(target_object_id)
        ee_to_world = self.ri.get_pose("tipLink")
        obj_to_ee = pp.multiply(pp.invert(ee_to_world), obj_to_world)

        self.ri.attachments = [
            pp.Attachment(
                self.ri.robot, self.ri.ee, obj_to_ee, target_object_id
            )
        ]

        # ---------------------------------------------------------------------

        self.object_ids = object_ids
        self.target_object_id = target_object_id
        self.target_object_class = data["class_id"][target_index]
        self.target_object_visibility = data["visibility"][target_index]

        self.i = 0
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
        ee_pose = np.hstack(self.ri.get_pose("tipLink")).astype(np.float32)
        grasp_flags, object_labels, object_poses = self.get_object_state(
            self.object_ids, self.target_object_id
        )
        obs = dict(
            ee_pose=ee_pose,
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
        dx, dy, dz, da, db, dg = self.actions[act_result.action[0]]

        assert self.target_object_id == self.ri.attachments[0].child

        c = mercury.geometry.Coordinate(
            *mercury.pybullet.get_pose(self.target_object_id)
        )
        c.translate([dx, dy, dz], wrt="world")
        c.rotate([da, db, dg], wrt="world")

        with pp.LockRenderer(), pp.WorldSaver():
            with self.ri.enabling_attachments():
                j = self.ri.solve_ik(
                    c.pose,
                    move_target=self.ri.robot_model.attachment_link0,
                    n_init=1,
                )
            if j is None:
                return

            if not self.ri.validatej(j, obstacles=[self.plane]):
                return

            return j

    def step(self, act_result):
        if not hasattr(act_result, "j"):
            act_result.j = self.validate_action(act_result)
        j = act_result.j
        terminate = act_result.action[1]

        if j is None:
            reward = 0
            if not self.eval:
                if self._use_reward_translation:
                    reward += -1
                if self._use_reward_max_velocity:
                    reward += -3
            return Transition(
                observation=self.get_obs(),
                reward=reward,
                terminal=True,
                info=dict(needs_reset=False, is_invalid=True),
            )

        with np.printoptions(precision=2):
            logger.info(
                f"[{act_result.action}] "
                f"{np.array(self.actions[act_result.action[0]])}"
                f"{act_result.action[1]}"
            )

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

        for _ in self.ri.movej(j, speed=self._speed):
            pp.step_simulation()
            step_callback()
            if self._gui:
                time.sleep(pp.get_time_step())

        if terminate:
            for _ in self.ri.movej(self.ri.homej, speed=self._speed):
                pp.step_simulation()
                step_callback()
                if self._gui:
                    time.sleep(pp.get_time_step())

        # ---------------------------------------------------------------------

        self.i += 1

        for object_id in self.object_ids:
            if object_id == self.target_object_id:
                continue
            self.translations[object_id] = (
                self.translations[object_id] + translations[object_id]
            )
            self.max_velocities[object_id] = max(
                self.max_velocities[object_id], max_velocities[object_id]
            )

        # primary task
        if terminate:
            terminal = True
            reward = 1
        elif self.i == self.episode_length:
            terminal = True
            reward = 0
        else:
            terminal = False
            reward = 0

        # secondary tasks
        if self._use_reward_translation:
            reward += -sum(translations.values())
        if self._use_reward_max_velocity and terminal:
            reward += -sum(self.max_velocities.values())

        logger.info(f"Reward={reward:.2f}, Terminal={terminal}")

        info = {
            "translation": sum(translations.values()),
            "needs_reset": terminal,
            "is_invalid": False,
        }
        if terminal:
            info["max_velocity"] = sum(self.max_velocities.values())

        return Transition(
            observation=self.get_obs(),
            reward=reward,
            terminal=terminal,
            info=info,
        )
