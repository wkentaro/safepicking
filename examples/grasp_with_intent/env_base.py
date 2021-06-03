import copy
import time

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


class EnvBase(Env):

    # parameters
    IMAGE_HEIGHT = 240
    IMAGE_WIDTH = 240

    PILES_DIR = home / "data/mercury/pile_generation"
    PILE_TRAIN_IDS = np.arange(0, 1000)
    PILE_EVAL_IDS = np.arange(1000, 1200)
    PILE_POSITION = np.array([0.5, 0, 0])

    c = mercury.geometry.Coordinate([0, 0.8, 0.8])
    c.rotate([np.pi / 2, 0, 0])
    BIN_EXTENTS = (0.3, 0.4, 0.2)
    BIN_POSE = c.pose

    CAMERA_POSITION = np.array([PILE_POSITION[0], PILE_POSITION[1], 0.7])

    PRE_PLACE_POSE = BIN_POSE[0] + (0, -0.3, 0), (0, 0, 0, 1)
    PLACE_POSE = BIN_POSE[0], PRE_PLACE_POSE[1]

    def __init__(
        self, class_ids, gui=True, retime=1, step_callback=None, mp4=None
    ):
        super().__init__()

        self._class_ids = class_ids
        self._gui = gui
        self._retime = retime
        self._step_callback = step_callback
        self._mp4 = mp4

        self.random_state = np.random.RandomState()

    def env(self):
        return

    @property
    def observation_elements(self):
        elements = []
        for name, space in self.observation_space.spaces.items():
            elements.append(ObservationElement(name, space.shape, space.dtype))
        return elements

    def shutdown(self):
        pp.disconnect()

    def launch(self):
        pp.connect(use_gui=self._gui, mp4=self._mp4)
        pp.add_data_path()

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
            height=self.IMAGE_HEIGHT,
            width=self.IMAGE_WIDTH,
        )

        data = dict(np.load(pile_file))

        PILE_AABB = (
            self.PILE_POSITION + [-0.25, -0.25, -0.05],
            self.PILE_POSITION + [0.25, 0.25, 0.5],
        )
        pp.draw_aabb(PILE_AABB)

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

        mercury.pybullet.duplicate(
            self.fg_object_id,
            collision=False,
            rgba_color=(1, 1, 1, 0.5),
            position=self.PLACE_POSE[0],
            quaternion=self.PLACE_POSE[1],
            mesh_scale=(0.95, 0.95, 0.95),
        )

        self.bin = mercury.pybullet.create_bin(*self.BIN_EXTENTS)
        pp.set_pose(self.bin, self.BIN_POSE)

        for _ in range(int(1 / pp.get_time_step())):
            p.stepSimulation()
            if self._step_callback:
                self._step_callback()
            if self._gui:
                time.sleep(pp.get_time_step())

        self.setj_to_camera_pose()
        self.update_obs()

        return self.get_obs()

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
