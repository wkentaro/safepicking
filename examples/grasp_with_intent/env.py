import copy
import time

import gym
import imgviz
from loguru import logger
import numpy as np
import path
import pybullet as p
import pybullet_planning as pp

from yarr.envs.env import Env
from yarr.utils.observation_type import ObservationElement
from yarr.utils.transition import Transition

import mercury


home = path.Path("~").expanduser()


class GraspWithIntentEnv(Env):

    piles_dir = home / "data/mercury/pile_generation"

    def __init__(self, gui=True, retime=1, step_callback=None):
        super().__init__()

        self._gui = gui
        self._retime = retime
        self._step_callback = step_callback

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
        ins = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3, 240, 240),
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
                ins=ins,
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

    def reset(self, pile_file=None):
        raise_on_failure = pile_file is not None

        if pile_file is None:
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
            suction_max_force=None,
            suction_surface_threshold=np.inf,
            suction_surface_alignment=False,
            planner="RRTConnect",
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

        fg_class_id = 11
        if fg_class_id not in data["class_id"]:
            if raise_on_failure:
                raise RuntimeError("fg_object_ids == []")
            else:
                return self.reset()

        num_instances = len(data["class_id"])
        object_ids = []
        fg_object_ids = []
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
                pp.draw_pose(
                    ([0, 0, 0], [0, 0, 0, 1]),
                    parent=object_id,
                    width=2,
                    length=0.2,
                )
            object_ids.append(object_id)
            if class_id == fg_class_id:
                fg_object_ids.append(object_id)
        self.object_ids = object_ids
        self.fg_object_ids = fg_object_ids

        # self.bin = mercury.pybullet.create_bin(0.3, 0.2, 0.2)
        # c = mercury.geometry.Coordinate()
        # c.position = [0, 0.5, 0.5]
        # c.rotate([0, np.deg2rad(90), 0])
        # c.rotate([np.deg2rad(90), 0, 0])
        # pp.set_pose(self.bin, c.pose)

        for _ in range(240):
            p.stepSimulation()
            if self._step_callback:
                self._step_callback()
            if self._gui:
                time.sleep(pp.get_time_step())

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
                c.pose, move_target=self.ri.robot_model.camera_link
            )
        self.ri.setj(j)

    def update_obs(self):
        rgb, depth, segm = self.ri.get_camera_image()
        fg_mask = np.isin(segm, self.fg_object_ids)
        K = self.ri.get_opengl_intrinsic_matrix()
        pcd = mercury.geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )
        c_camera_to_world = mercury.geometry.Coordinate(
            *self.ri.get_pose("camera_link")
        )
        pcd = mercury.geometry.transform_points(pcd, c_camera_to_world.matrix)
        ins = np.zeros_like(pcd)
        for obj in self.object_ids:
            ins[segm == obj] = pcd[segm == obj] - pp.get_pose(obj)[0]
        ins = imgviz.normalize(
            ins, min_value=(-0.15,) * 3, max_value=(0.15,) * 3
        )
        self.obs = dict(
            rgb=rgb.transpose(2, 0, 1),
            depth=depth,
            ins=ins.transpose(2, 0, 1).astype(np.float32),
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
        reward = self._step(act_result)

        self.setj_to_camera_pose()
        self.update_obs()

        return Transition(
            observation=self.get_obs(),
            reward=reward,
            terminal=True,
            info=dict(needs_reset=True),
        )

    def validate_action(self, act_result):
        logger.info(f"Validating action: {act_result.action}")

        lock_renderer = pp.LockRenderer()
        world_saver = pp.WorldSaver()

        def before_return():
            self.ri.attachments = []
            world_saver.restore()
            lock_renderer.restore()

        result = {}

        y, x = act_result.action

        is_fg = self.obs["fg_mask"][y, x]
        if not is_fg:
            logger.error(f"non fg area is selected: {act_result.action}")
            before_return()
            return False, result

        object_id = self.obs["segm"][y, x]
        result["object_id"] = object_id
        if object_id not in self.object_ids:
            logger.error(
                f"object {object_id} is not in the graspable objects: "
                f"{self.object_ids}"
            )
            before_return()
            return False, result

        K = self.ri.get_opengl_intrinsic_matrix()
        pcd_in_camera = mercury.geometry.pointcloud_from_depth(
            self.obs["depth"], fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )

        camera_to_world = self.ri.get_pose("camera_link")
        ee_to_world = self.ri.get_pose("tipLink")
        camera_to_ee = pp.multiply(pp.invert(ee_to_world), camera_to_world)
        pcd_in_ee = mercury.geometry.transform_points(
            pcd_in_camera,
            mercury.geometry.transformation_matrix(*camera_to_ee),
        )

        normals = mercury.geometry.normals_from_pointcloud(pcd_in_ee)

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

        obj_to_world = pp.get_pose(object_id)

        j = self.ri.solve_ik(c.pose)
        if j is None:
            logger.error(
                f"Failed to solve pre-grasping IK: {act_result.action}"
            )
            before_return()
            return False, result
        result["j_pre_grasp"] = j

        self.ri.setj(j)

        c.translate([0, 0, 0.1])
        j = self.ri.solve_ik(c.pose)
        if j is None:
            logger.error(f"Failed to solve grasping IK: {act_result.action}")
            before_return()
            return False, result
        result["j_grasp"] = j

        self.ri.setj(j)

        ee_to_world = self.ri.get_pose("tipLink")
        obj_to_ee = pp.multiply(pp.invert(ee_to_world), obj_to_world)
        self.ri.attachments = [
            pp.Attachment(self.ri.robot, self.ri.ee, obj_to_ee, object_id)
        ]

        self.ri.setj(self.ri.homej)
        self.ri.attachments[0].assign()

        with self.ri.enabling_attachments():
            obj_to_world = [0, 0.7, 0.15], [0, 0, 0, 1]
            pp.draw_pose(obj_to_world)

            j = self.ri.solve_ik(
                obj_to_world,
                move_target=self.ri.robot_model.attachment_link0,
            )
        if j is None:
            logger.error(f"Failed to solve placing IK: {act_result.action}")
            before_return()
            return False, result
        result["j_place"] = j

        self.ri.setj(j)
        self.ri.attachments[0].assign()

        path = self.ri.planj(j, obstacles=[self.plane])
        if path is None:
            logger.error(f"Goal state is invalid: {act_result.action}")
            before_return()
            return False, result

        before_return()
        return True, result

    def _step(self, act_result):
        if hasattr(act_result, "validation_result"):
            is_valid = True
            validation_result = act_result.validation_result
        else:
            is_valid, validation_result = self.validate_action(act_result)

        if not is_valid:
            logger.error(f"Invalid action: {act_result.action}")
            return 0

        assert len(self.ri.attachments) == 0
        assert self.ri.gripper.grasped_object is None
        assert self.ri.gripper.activated is False

        object_id = validation_result["object_id"]
        obj_to_world = pp.get_pose(object_id)

        self.ri.setj(validation_result["j_pre_grasp"])
        if self._gui:
            time.sleep(1)

        self.ri.setj(validation_result["j_grasp"])
        if self._gui:
            time.sleep(1)

        ee_to_world = self.ri.get_pose("tipLink")
        obj_to_ee = pp.multiply(pp.invert(ee_to_world), obj_to_world)
        self.ri.attachments = [
            pp.Attachment(self.ri.robot, self.ri.ee, obj_to_ee, object_id)
        ]

        self.ri.setj(self.ri.homej)
        self.ri.attachments[0].assign()
        if self._gui:
            time.sleep(1)

        self.ri.setj(validation_result["j_place"])
        self.ri.attachments[0].assign()
        if self._gui:
            time.sleep(1)

        def before_return():
            self.ri.ungrasp()
            pp.remove_body(object_id)
            self.object_ids.remove(object_id)

            for _ in range(240):
                p.stepSimulation()
                if self._step_callback:
                    self._step_callback()
                if self._gui:
                    time.sleep(pp.get_time_step())

        before_return()
        return 1
