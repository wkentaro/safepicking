import time

import gym
from loguru import logger
import numpy as np
import path
import pybullet as p
import pybullet_planning as pp

import mercury

from env_base import EnvBase


home = path.Path("~").expanduser()


class PickAndPlaceEnv(EnvBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        rgb = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, self.IMAGE_HEIGHT, self.IMAGE_WIDTH),
            dtype=np.uint8,
        )
        depth = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH),
            dtype=np.float32,
        )
        ocs = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3, self.IMAGE_HEIGHT, self.IMAGE_WIDTH),
            dtype=np.float32,
        )
        fg_mask = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH),
            dtype=np.uint8,
        )
        self.observation_space = gym.spaces.Dict(
            dict(
                rgb=rgb,
                depth=depth,
                ocs=ocs,
                fg_mask=fg_mask,
            ),
        )

    @property
    def action_shape(self):
        return (2,)

    def update_obs(self):
        rgb, depth, segm = self.ri.get_camera_image()
        fg_mask = segm == self.fg_object_id
        K = self.ri.get_opengl_intrinsic_matrix()
        pcd_in_camera = mercury.geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )
        c_camera_to_world = mercury.geometry.Coordinate(
            *self.ri.get_pose("camera_link")
        )
        pcd_in_world = mercury.geometry.transform_points(
            pcd_in_camera, c_camera_to_world.matrix
        )
        ocs = np.zeros_like(pcd_in_world)
        for obj in self.object_ids:
            world_to_obj = pp.invert(pp.get_pose(obj))
            ocs[segm == obj] = mercury.geometry.transform_points(
                pcd_in_world,
                mercury.geometry.transformation_matrix(*world_to_obj),
            )[segm == obj]
        self.obs = dict(
            rgb=rgb.transpose(2, 0, 1),
            depth=depth,
            ocs=ocs.transpose(2, 0, 1).astype(np.float32),
            fg_mask=fg_mask.astype(np.uint8),
            segm=segm,
            camera_to_world=np.hstack(c_camera_to_world.pose),
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

        camera_to_world = (
            self.obs["camera_to_world"][:3],
            self.obs["camera_to_world"][3:],
        )
        ee_to_world = self.ri.get_pose("tipLink")
        camera_to_ee = pp.multiply(pp.invert(ee_to_world), camera_to_world)
        pcd_in_ee = mercury.geometry.transform_points(
            pcd_in_camera,
            mercury.geometry.transformation_matrix(*camera_to_ee),
        )

        normals = mercury.geometry.normals_from_pointcloud(pcd_in_ee)

        position = pcd_in_ee[y, x]
        quaternion = mercury.geometry.quaternion_from_vec2vec(
            [0, 0, 1], normals[y, x], flip=False
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

        j = self.ri.solve_ik(c.pose, rotation_axis="z")
        if j is not None and not self.ri.validatej(
            j, obstacles=[self.plane, self.bin] + self.object_ids
        ):
            j = None
        if j is None:
            logger.error(
                f"Failed to solve pre-grasping IK: {act_result.action}"
            )
            before_return()
            return False, result
        result["j_pre_grasp"] = j

        js = self.ri.planj(
            j, obstacles=[self.plane, self.bin] + self.object_ids
        )
        if js is None:
            logger.error(
                f"Failed to solve pre-grasping path: {act_result.action}"
            )
            before_return()
            return False, result
        result["js_pre_grasp"] = js

        self.ri.setj(j)

        c = mercury.geometry.Coordinate(*self.ri.get_pose("tipLink"))
        c.translate([0, 0, 0.1])
        j = self.ri.solve_ik(c.pose, rotation_axis=True)
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

        j = self.ri.homej
        self.ri.setj(j)
        self.ri.attachments[0].assign()

        with self.ri.enabling_attachments():
            j = self.ri.solve_ik(
                self.PRE_PLACE_POSE,
                move_target=self.ri.robot_model.attachment_link0,
            )
        if j is None:
            logger.error(
                f"Failed to solve pre-placing IK: {act_result.action}"
            )
            before_return()
            return False, result
        result["j_pre_place"] = j

        obstacles = [self.plane, self.bin] + self.object_ids
        obstacles.remove(self.fg_object_id)
        assert self.ri.attachments[0].child == self.fg_object_id
        js = self.ri.planj(
            j,
            obstacles=obstacles,
            min_distances_start_goal={
                (self.ri.attachments[0].child, -1): -0.01
            },
        )
        if js is None:
            logger.error(
                f"Failed to solve pre-placing path: {act_result.action}"
            )
            before_return()
            return False, result
        result["js_pre_place"] = js

        self.ri.setj(j)
        self.ri.attachments[0].assign()

        js = []
        with self.ri.enabling_attachments():
            for pose in pp.interpolate_poses(
                pose1=self.PRE_PLACE_POSE, pose2=self.PLACE_POSE
            ):
                j = self.ri.solve_ik(
                    pose,
                    move_target=self.ri.robot_model.attachment_link0,
                    n_init=1,
                    thre=0.01,
                    rthre=np.deg2rad(10),
                )
                if j is None:
                    logger.error(
                        f"Failed to solve placing IK: {act_result.action}"
                    )
                    before_return()
                    return False, result
                is_valid = self.ri.validatej(
                    j,
                    obstacles=[self.plane, self.bin],
                    min_distances={(self.ri.attachments[0].child, -1): -0.01},
                )
                if not is_valid:
                    logger.error(
                        f"Failed to solve placing path: {act_result.action}"
                    )
                    before_return()
                    return False, result
                js.append(j)
        result["js_place"] = js

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

        js = validation_result["js_pre_grasp"]
        for _ in (_ for j in js for _ in self.ri.movej(j)):
            pp.step_simulation()
            time.sleep(pp.get_time_step())

        object_id = validation_result["object_id"]
        obj_to_world = pp.get_pose(object_id)

        for _ in self.ri.grasp(min_dz=0.1, max_dz=0.15, speed=0.001):
            pp.step_simulation()
            time.sleep(pp.get_time_step())
            if self.ri.gripper.detect_contact():
                break
        self.ri.gripper.activate()

        ee_to_world = self.ri.get_pose("tipLink")
        obj_to_ee = pp.multiply(pp.invert(ee_to_world), obj_to_world)
        self.ri.attachments = [
            pp.Attachment(self.ri.robot, self.ri.ee, obj_to_ee, object_id)
        ]

        for _ in (_ for _ in self.ri.movej(self.ri.homej)):
            pp.step_simulation()
            time.sleep(pp.get_time_step())

        js = validation_result["js_pre_place"]
        for _ in (_ for j in js for _ in self.ri.movej(j)):
            pp.step_simulation()
            time.sleep(pp.get_time_step())

        js = validation_result["js_place"]
        for _ in (_ for j in js for _ in self.ri.movej(j)):
            pp.step_simulation()
            time.sleep(pp.get_time_step())

        def before_return():
            for _ in range(240):
                p.stepSimulation()
                if self._step_callback:
                    self._step_callback()
                if self._gui:
                    time.sleep(pp.get_time_step())

            self.ri.ungrasp()

            for _ in range(240):
                p.stepSimulation()
                if self._step_callback:
                    self._step_callback()
                if self._gui:
                    time.sleep(pp.get_time_step())

            pp.remove_body(object_id)
            self.object_ids.remove(object_id)

        before_return()
        return 1
