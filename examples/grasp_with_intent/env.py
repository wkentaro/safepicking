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

import common_utils


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
                pp.draw_pose(
                    ([0, 0, 0], [0, 0, 0, 1]),
                    parent=object_id,
                    width=2,
                    length=0.2,
                )
            object_ids.append(object_id)
        self.object_ids = object_ids

        for _ in range(240):
            p.stepSimulation()
            if self._step_callback:
                self._step_callback()
            if self._gui:
                time.sleep(pp.get_time_step())

        self.setj_to_camera_pose()
        self.update_obs()

        self._prev_step_num_objects = None

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
        fg_mask = ~np.isin(segm, [-1, self.plane, self.ri.robot])
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

        needs_reset = (
            len(self.object_ids) == 0
            or self.obs["fg_mask"].sum() == 0
            or (
                not self.eval
                and self._prev_step_num_objects == len(self.object_ids)
                and np.random.random() < 0.1
            )
        )

        self._prev_step_num_objects = len(self.object_ids)

        return Transition(
            observation=self.get_obs(),
            reward=reward,
            terminal=True,
            info=dict(needs_reset=needs_reset),
        )

    def validate_action(self, act_result):
        logger.info(f"Validating action: {act_result.action}")

        lock_renderer = pp.LockRenderer()
        world_saver = pp.WorldSaver()

        def before_return():
            self.ri.attachments = []
            world_saver.restore()
            lock_renderer.restore()

        data = {}

        y, x = act_result.action

        object_id = self.obs["segm"][y, x]
        data["object_id"] = object_id
        if object_id not in self.object_ids:
            logger.error(
                f"object {object_id} is not in the graspable objects: "
                f"{self.object_ids}"
            )
            before_return()
            return False, data

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
            before_return()
            return False, data
        data["c_grasp"] = c

        for i in range(3):
            j = self.ri.solve_ik(c.pose)
            if j is None:
                if i == 2:
                    logger.error(
                        "pre-grasping ik solution is not found: "
                        f"{act_result.action}"
                    )
                    before_return()
                    return False, data
                else:
                    continue

            path = self.ri.planj(j, obstacles=[self.plane] + self.object_ids)
            if path is None:
                if i == 2:
                    logger.error(
                        f"pre-grasping path is not found: {act_result.action}"
                    )
                    before_return()
                    return False, data
                else:
                    continue
        data["path_grasp"] = path

        self.ri.setj(path[-1])

        c = mercury.geometry.Coordinate(*self.ri.get_pose("tipLink"))
        c.translate([0, 0, 0.1])

        j = self.ri.solve_ik(c.pose)
        if j is None:
            logger.error(
                f"grasping ik solution is not found: {act_result.action}"
            )
            before_return()
            return False, data
        self.ri.setj(j)

        obj_to_world = pp.get_pose(object_id)
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

        self.ri.setj(self.ri.homej)
        self.ri.attachments[0].assign()

        c_place = mercury.geometry.Coordinate(
            [0, 0.5, 0],
            common_utils.get_canonical_quaternion(
                common_utils.get_class_id(object_id)
            ),
        )
        with pp.LockRenderer(), pp.WorldSaver():
            pp.set_pose(object_id, c_place.pose)
            c_place.position[2] -= pp.get_aabb(object_id).lower[2]
            c_place.position[2] += 0.01

        for i in range(3):
            with self.ri.enabling_attachments():
                obj_to_world = self.ri.get_pose("attachment_link0")

                move_target_to_world = mercury.geometry.Coordinate(
                    *obj_to_world
                )
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
                )

            if j is None:
                if i == 2:
                    logger.error("placing ik solution is not found")
                    before_return()
                    return False, data
                else:
                    continue

            obstacles = [self.plane] + self.object_ids
            obstacles.remove(self.ri.attachments[0].child)
            path = self.ri.planj(j, obstacles=obstacles)
            if path is None:
                if i == 2:
                    logger.error("placing path is not found")
                    before_return()
                    return False, data
                else:
                    continue
        data["path_place"] = path

        before_return()
        return True, data

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

        path = validation_result["path_grasp"]
        for _ in (_ for j in path for _ in self.ri.movej(j)):
            p.stepSimulation()
            if self._step_callback:
                self._step_callback()
            if self._gui:
                time.sleep(pp.get_time_step())

        try:
            for _ in self.ri.grasp(
                min_dz=0.08, max_dz=0.12, speed=0.005, rotation_axis=True
            ):
                p.stepSimulation()
                if self._step_callback:
                    self._step_callback()
                if self._gui:
                    time.sleep(pp.get_time_step())
        except RuntimeError as e:
            logger.error(e)
            self.ri.ungrasp()
            return 0

        if not self.ri.gripper.check_grasp():
            logger.error("grasping is failed")
            self.ri.ungrasp()
            return 0

        grasped_object = validation_result["object_id"]

        def before_return():
            self.ri.ungrasp()
            pp.remove_body(grasped_object)
            self.object_ids.remove(grasped_object)

            for _ in range(240):
                p.stepSimulation()
                if self._step_callback:
                    self._step_callback()
                if self._gui:
                    time.sleep(pp.get_time_step())

        ee_to_world = self.ri.get_pose("tipLink")
        obj_to_ee = pp.multiply(pp.invert(ee_to_world), obj_to_world)
        self.ri.attachments = [
            pp.Attachment(
                parent=self.ri.robot,
                parent_link=self.ri.ee,
                grasp_pose=obj_to_ee,
                child=grasped_object,
            )
        ]

        for _ in self.ri.movej(self.ri.homej):
            p.stepSimulation()
            if self._step_callback:
                self._step_callback()
            if self._gui:
                time.sleep(pp.get_time_step())

        path = validation_result["path_place"]
        for _ in (_ for j in path for _ in self.ri.movej(j)):
            p.stepSimulation()
            if self._step_callback:
                self._step_callback()
            if self._gui:
                time.sleep(pp.get_time_step())

        for _ in range(240):
            p.stepSimulation()
            if self._step_callback:
                self._step_callback()
            if self._gui:
                time.sleep(pp.get_time_step())

        self.ri.ungrasp()

        before_return()
        return 1
