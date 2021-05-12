import gym
import numpy as np
import path
import pybullet_planning as pp

from env_base import EnvBase


home = path.Path("~").expanduser()


class ReorientEnv(EnvBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        object_poses = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8, 7),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            dict(
                object_poses=object_poses,
            ),
        )

    @property
    def action_shape(self):
        return ()

    def update_obs(self):
        rgb, depth, segm = self.ri.get_camera_image()
        object_poses = np.zeros((8, 7), dtype=np.float32)
        for i, object_id in enumerate(self.object_ids):
            object_poses[i] = np.concatenate(pp.get_pose(object_id))
        self.obs = dict(
            rgb=rgb.transpose(2, 0, 1),
            depth=depth,
            segm=segm,
            object_poses=object_poses,
        )
