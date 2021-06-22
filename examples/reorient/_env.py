import time

# import imgviz
from loguru import logger
import numpy as np
import path
import pybullet as p
import pybullet_planning as pp

import mercury

import _utils
from legacy.init_place_scene import init_place_scene


home = path.Path("~").expanduser()


class Env:

    # parameters
    IMAGE_HEIGHT = 240
    IMAGE_WIDTH = 240

    PILES_DIR = home / "data/mercury/pile_generation"
    PILE_TRAIN_IDS = np.arange(0, 1000)
    PILE_EVAL_IDS = np.arange(1000, 1200)
    PILE_POSITION = np.array([0.5, 0, 0])

    CAMERA_POSITION = np.array([PILE_POSITION[0], PILE_POSITION[1], 0.7])

    @property
    def PLACE_POSE(self):
        return self._place_pose

    @property
    def PRE_PLACE_POSE(self):
        c = mercury.geometry.Coordinate(*self.PLACE_POSE)
        c.translate([0, -0.3, 0], wrt="world")
        return c.pose

    def __init__(
        self, class_ids, gui=True, retime=1, step_callback=None, mp4=None
    ):
        super().__init__()

        self._class_ids = class_ids
        self._gui = gui
        self._retime = retime
        self._step_callback = step_callback
        self._mp4 = mp4

        self.eval = False
        self.random_state = np.random.RandomState()

    def shutdown(self):
        pp.disconnect()

    def launch(self):
        pp.connect(use_gui=self._gui, mp4=self._mp4)
        pp.add_data_path()

    @property
    def bg_objects(self):
        return [self.plane] + self.containers

    def reset(self, pile_file=None):
        raise_on_error = pile_file is not None

        if pile_file is None:
            if self.eval:
                i = self.random_state.choice(self.PILE_EVAL_IDS)
            else:
                i = self.random_state.choice(self.PILE_TRAIN_IDS)
            pile_file = self.PILES_DIR / f"{i:08d}.npz"

        if not pp.is_connected():
            self.launch()

        pp.reset_simulation()
        pp.enable_gravity()
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=90,
            cameraPitch=-60,
            cameraTargetPosition=(0, 0, 0),
        )
        with pp.LockRenderer():
            self.plane = pp.load_pybullet("plane.urdf")
            pp.set_texture(self.plane)
            pp.set_color(self.plane, (100 / 256, 100 / 256, 100 / 256, 1))

        self.ri = mercury.pybullet.PandaRobotInterface(
            suction_max_force=None,
            suction_surface_threshold=np.inf,
            suction_surface_alignment=False,
            # SBL gives larger margin from obstacles than RRTConnect
            # planner="SBL",
            planner="RRTConnect",
        )
        c_cam_to_ee = mercury.geometry.Coordinate()
        c_cam_to_ee.translate([0, -0.1, -0.1])
        self.ri.add_camera(
            pose=c_cam_to_ee.pose,
            fovy=np.deg2rad(60),
            height=self.IMAGE_HEIGHT,
            width=self.IMAGE_WIDTH,
        )

        if 0:
            sphere = pp.create_sphere(
                0.8, color=(1, 0, 0, 0.2), collision=False
            )
            pp.set_pose(sphere, ([0, 0, 0.1], [0, 0, 0, 1]))
            sphere = pp.create_sphere(
                0.8, color=(1, 0, 0, 0.2), collision=False
            )
            pp.set_pose(sphere, ([0, 0.3, 0.3], [0, 0, 0, 1]))

        data = dict(np.load(pile_file))

        PILE_AABB = (
            self.PILE_POSITION + [-0.25, -0.25, -0.05],
            self.PILE_POSITION + [0.25, 0.25, 0.5],
        )
        # pp.draw_aabb(PILE_AABB)
        box = pp.create_box(
            w=PILE_AABB[1][0] - PILE_AABB[0][0],
            l=PILE_AABB[1][1] - PILE_AABB[0][1],
            h=0.01,
            color=(0, 100 / 256, 0, 1),
            collision=False,
        )
        pp.set_pose(box, (self.PILE_POSITION, [0, 0, 0, 1]))

        num_instances = len(data["class_id"])
        object_ids = []
        fg_object_ids = []
        for i in range(num_instances):
            class_id = data["class_id"][i]
            position = data["position"][i]
            quaternion = data["quaternion"][i]

            position += self.PILE_POSITION

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
            pp.set_dynamics(object_id, lateralFriction=0.7)
            object_ids.append(object_id)

            contained = pp.aabb_contains_aabb(
                pp.get_aabb(object_id), PILE_AABB
            )

            if class_id in self._class_ids and visibility > 0.95 and contained:
                fg_object_ids.append(object_id)

        if not fg_object_ids:
            if raise_on_error:
                raise RuntimeError
            else:
                return self.reset()

        self.object_ids = object_ids
        self.fg_object_id = self.random_state.choice(fg_object_ids)

        pp.draw_aabb(
            pp.get_aabb(self.fg_object_id),
            color=(1, 0, 0),
            width=2,
        )

        # create container
        self.containers, self._place_pose = init_place_scene(
            class_id=_utils.get_class_id(self.fg_object_id)
        )

        for _ in range(int(1 / pp.get_time_step())):
            p.stepSimulation()
            if self._step_callback:
                self._step_callback()
            if self._gui:
                time.sleep(pp.get_time_step())

        self.setj_to_camera_pose()
        self.update_obs()

    def setj_to_camera_pose(self):
        self.ri.setj(self.ri.homej)
        j = None
        while j is None:
            c = mercury.geometry.Coordinate(*self.ri.get_pose("camera_link"))
            c.position = self.CAMERA_POSITION
            j = self.ri.solve_ik(
                c.pose, move_target=self.ri.robot_model.camera_link
            )
        self.ri.setj(j)

    def update_obs(self):
        rgb, depth, segm = self.ri.get_camera_image()
        # if pp.has_gui():
        #     imgviz.io.cv_imshow(
        #         np.hstack((rgb, imgviz.depth2rgb(depth))), "update_obs"
        #     )
        #     imgviz.io.cv_waitkey(100)
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

        j = self.ri.solve_ik(c.pose, rotation_axis="z")
        if j is None:
            logger.error(
                f"Failed to solve pre-grasping IK: {act_result.action}"
            )
            before_return()
            return False, result
        if not self.ri.validatej(
            j, obstacles=self.bg_objects + self.object_ids
        ):
            logger.error(f"j_pre_grasp is invalid: {act_result.action}")
            before_return()
            return False, result
        result["j_pre_grasp"] = j

        js = self.ri.planj(j, obstacles=self.bg_objects + self.object_ids)
        if js is None:
            logger.error(
                f"Failed to solve pre-grasping path: {act_result.action}"
            )
            before_return()
            return False, result
        result["js_pre_grasp"] = js

        self.ri.setj(j)

        c = mercury.geometry.Coordinate(*self.ri.get_pose("tipLink"))
        obstacles = self.bg_objects + self.object_ids
        obstacles.remove(self.fg_object_id)
        for _ in range(10):
            c.translate([0, 0, 0.01])
            j = self.ri.solve_ik(c.pose, rotation_axis=True)
            if j is None:
                logger.error(
                    f"Failed to solve grasping IK: {act_result.action}"
                )
                before_return()
                return False, result
            if not self.ri.validatej(j, obstacles=obstacles):
                logger.error(f"grasping j is invalid: {act_result.action}")
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
            j = self.ri.solve_ik(
                self.PRE_PLACE_POSE,
                move_target=self.ri.robot_model.attachment_link0,
                thre=0.03,
                rthre=np.deg2rad(30),
            )
        if j is None:
            logger.error(
                f"Failed to solve pre-placing IK: {act_result.action}"
            )
            before_return()
            return False, result
        result["j_pre_place"] = j

        self.ri.setj(j)
        self.ri.attachments[0].assign()

        with self.ri.enabling_attachments():
            j = self.ri.solve_ik(
                self.PLACE_POSE,
                move_target=self.ri.robot_model.attachment_link0,
            )
        if j is None:
            logger.error(f"Failed to solve placing IK: {act_result.action}")
            before_return()
            return False, result
        result["j_place"] = j

        obstacles = self.bg_objects + self.object_ids
        obstacles.remove(self.fg_object_id)

        if not self.ri.validatej(result["j_place"], obstacles=obstacles):
            logger.error(f"j_place is invalid: {act_result.action}")
            before_return()
            return False, result

        assert self.ri.attachments[0].child == self.fg_object_id

        self.ri.setj(self.ri.homej)
        self.ri.attachments[0].assign()

        js = self.ri.planj(
            result["j_pre_place"],
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

        self.ri.setj(js[-1])
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
                    obstacles=self.bg_objects,
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

    def execute(self, validation_result):
        assert len(self.ri.attachments) == 0
        assert self.ri.gripper.grasped_object is None
        assert self.ri.gripper.activated is False

        js = validation_result["js_pre_grasp"]
        for _ in (_ for j in js for _ in self.ri.movej(j)):
            pp.step_simulation()
            time.sleep(pp.get_time_step())

        object_id = validation_result["object_id"]
        obj_to_world = pp.get_pose(object_id)

        for _ in self.ri.grasp(
            min_dz=0.1, max_dz=0.15, speed=0.001, rotation_axis=True
        ):
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

        for _ in self.ri.movej(self.ri.homej, speed=0.005):
            pp.step_simulation()
            time.sleep(pp.get_time_step())

        js = validation_result["js_pre_place"]
        for _ in (_ for j in js for _ in self.ri.movej(j)):
            pp.step_simulation()
            time.sleep(pp.get_time_step())

        js = validation_result["js_place"]
        for _ in (_ for j in js for _ in self.ri.movej(j, speed=0.005)):
            pp.step_simulation()
            time.sleep(pp.get_time_step())

        for _ in range(240):
            p.stepSimulation()
            if self._step_callback:
                self._step_callback()
            if self._gui:
                time.sleep(pp.get_time_step())

        self.ri.ungrasp()

        for _ in (_ for j in js[::-1] for _ in self.ri.movej(j, speed=0.005)):
            pp.step_simulation()
            time.sleep(pp.get_time_step())

        for _ in self.ri.movej(self.ri.homej):
            pp.step_simulation()
            time.sleep(pp.get_time_step())
