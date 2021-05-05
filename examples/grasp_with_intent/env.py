import copy
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


class GraspWithIntentEnv(Env):

    piles_dir = home / "data/mercury/pile_generation"

    def __init__(self, gui=True, retime=1):
        super().__init__()

        self._gui = gui
        self._retime = retime

        rgb = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, 240, 240),
            dtype=np.uint8,
        )
        depth = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(240, 240),
            dtype=np.float32,
        )
        fg_mask = gym.spaces.Box(
            low=0,
            high=1,
            shape=(240, 240),
            dtype=np.uint8,
        )
        self.observation_space = gym.spaces.Dict(
            dict(
                rgb=rgb,
                depth=depth,
                fg_mask=fg_mask,
            ),
        )

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

    def reset(self):
        if self.eval:
            i = np.random.randint(1000, 1200)
        else:
            i = np.random.randint(0, 1000)
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
            suction_max_force=None, planner="RRTConnect"
        )
        c_cam_to_ee = mercury.geometry.Coordinate()
        c_cam_to_ee.translate([0, -0.1, -0.1])
        self.ri.add_camera(
            pose=c_cam_to_ee.pose,
            fovy=np.deg2rad(60),
            height=240,
            width=240,
        )

        data = dict(np.load(pile_file))

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
                )
                pp.draw_pose(([0, 0, 0], [0, 0, 0, 1]), parent=object_id)
            object_ids.append(object_id)
        self.object_ids = object_ids

        self.setj_to_camera_pose()
        self.update_obs()

        return self.get_obs()

    def setj_to_camera_pose(self):
        aabb = [0.3, -0.2, 0], [0.7, 0.2, 0.6]
        pp.draw_aabb(aabb)

        self.ri.setj(self.ri.homej)
        j = None
        while j is None:
            c = mercury.geometry.Coordinate(*self.ri.get_pose("camera_link"))
            c.position = np.mean(aabb, axis=0)
            c.position[2] = 0.7
            j = self.ri.solve_ik(
                c.pose,
                move_target=self.ri.robot_model.camera_link,
                rotation_axis="z",
            )
        self.ri.setj(j)

    def update_obs(self):
        rgb, depth, segm = self.ri.get_camera_image()
        fg_mask = ~np.isin(segm, [-1, self.plane, self.ri.robot])
        self.obs = dict(
            rgb=rgb.transpose(2, 0, 1),
            depth=depth,
            fg_mask=fg_mask.astype(np.uint8),
            segm=segm,
        )

    def get_obs(self):
        obs = copy.deepcopy(self.obs)

        for key in list(obs.keys()):
            if key not in self.observation_space.spaces:
                obs.pop(key)

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

    def step(self, act_result):
        action = act_result.action
        reward = self._step(y=action[0], x=action[1])

        grasped_object = self.ri.gripper.grasped_object
        self.ri.ungrasp()
        if grasped_object:
            self.object_ids.remove(grasped_object)
            pp.remove_body(grasped_object)

        self.setj_to_camera_pose()
        self.update_obs()

        needs_reset = (
            len(self.object_ids) == 0
            or self.obs["fg_mask"].sum() == 0
            or np.random.random() < 0.1
        )

        return Transition(
            observation=self.get_obs(),
            reward=reward,
            terminal=True,
            info=dict(needs_reset=needs_reset),
        )

    def _step(self, y, x):
        segm = self.obs["segm"]
        depth = self.obs["depth"]
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

        object_id = segm[y, x]
        if object_id not in self.object_ids:
            logger.error(
                f"object {object_id} is not in the graspable objects: "
                f"{self.object_ids}"
            )
            return 0

        position = pcd_in_ee[y, x]
        quaternion = mercury.geometry.quaternion_from_vec2vec(
            [0, 0, 1], normals[y, x]
        )

        T_ee_to_ee_af_in_ee = mercury.geometry.transformation_matrix(
            position, quaternion
        )

        T_ee_to_world = mercury.geometry.transformation_matrix(
            *self.ri.get_pose("tipLink")
        )
        T_ee_to_ee = np.eye(4)
        T_ee_af_to_ee = T_ee_to_ee_af_in_ee @ T_ee_to_ee
        T_ee_af_to_world = T_ee_to_world @ T_ee_af_to_ee

        c = mercury.geometry.Coordinate(
            *mercury.geometry.pose_from_matrix(T_ee_af_to_world)
        )
        c.translate([0, 0, -0.1])

        max_angle = np.deg2rad(45)
        vec = mercury.geometry.transform_points(
            [[0, 0, 0], [0, 0, 1]], c.matrix
        )
        v0 = [0, 0, -1]
        v1 = vec[1] - vec[0]
        v1 /= np.linalg.norm(v1)
        angle = mercury.geometry.angle_between_vectors(v0, v1)
        if angle > max_angle:
            logger.error(
                f"angle ({np.rad2deg(angle):.1f} [deg]) > "
                f"{np.rad2deg(max_angle):.1f} [deg]"
            )
            return 0

        j = self.ri.solve_ik(c.pose)
        if j is None:
            return 0

        obj_to_world = pp.get_pose(object_id)

        path = self.ri.planj(j, obstacles=[self.plane] + self.object_ids)
        if path is None:
            return 0

        for _ in (_ for j in path for _ in self.ri.movej(j)):
            p.stepSimulation()
            if self._gui:
                time.sleep(pp.get_time_step())

        try:
            for _ in self.ri.grasp(min_dz=0.08, max_dz=0.12, speed=0.005):
                p.stepSimulation()
                if self._gui:
                    time.sleep(pp.get_time_step())
        except RuntimeError:
            return 0

        if not self.ri.gripper.check_grasp():
            return 0

        ee_to_world = self.ri.get_pose("tipLink")
        obj_to_ee = pp.multiply(pp.invert(ee_to_world), obj_to_world)
        self.ri.attachments = [
            pp.Attachment(
                parent=self.ri.robot,
                parent_link=self.ri.ee,
                grasp_pose=obj_to_ee,
                child=object_id,
            )
        ]

        for _ in self.ri.movej(self.ri.homej):
            p.stepSimulation()
            if self._gui:
                time.sleep(pp.get_time_step())

        c_place = mercury.geometry.Coordinate(
            [0, 0.5, 0],
            common_utils.get_canonical_quaternion(
                common_utils.get_class_id(object_id)
            ),
        )

        with pp.LockRenderer(), pp.WorldSaver():
            pp.set_pose(object_id, c_place.pose)
            c_place.position[2] -= pp.get_aabb(object_id).lower[2]

        with self.ri.enabling_attachments():
            obj_to_world = self.ri.get_pose("attachment_link0")

            move_target_to_world = mercury.geometry.Coordinate(*obj_to_world)
            move_target_to_world.transform(
                np.linalg.inv(
                    mercury.geometry.quaternion_matrix(c_place.quaternion)
                ),
                wrt="local",
            )
            move_target_to_world = move_target_to_world.pose

            ee_to_world = self.ri.get_pose("tipLink")
            move_target_to_ee = pp.multiply(
                pp.invert(ee_to_world), move_target_to_world
            )
            self.ri.add_link("move_target", pose=move_target_to_ee)

            j = self.ri.solve_ik(
                (c_place.position, [0, 0, 0, 1]),
                move_target=self.ri.robot_model.move_target,
                rotation_axis="z",
            )

        if j is None:
            return 0

        obstacles = [self.plane] + self.object_ids
        obstacles.remove(self.ri.attachments[0].child)
        path = self.ri.planj(j, obstacles=obstacles)
        if path is None:
            return 0

        for _ in (_ for j in path for _ in self.ri.movej(j)):
            p.stepSimulation()
            if self._gui:
                time.sleep(pp.get_time_step())

        for _ in range(240):
            p.stepSimulation()
            if self._gui:
                time.sleep(pp.get_time_step())

        self.ri.ungrasp()

        for _ in range(240):
            p.stepSimulation()
            if self._gui:
                time.sleep(pp.get_time_step())

        for _ in self.ri.move_to_homej([self.plane], self.object_ids):
            p.stepSimulation()
            if self._gui:
                time.sleep(pp.get_time_step())

        pp.remove_body(object_id)

        for _ in range(240):
            p.stepSimulation()
            if self._gui:
                time.sleep(pp.get_time_step())

        return 1
