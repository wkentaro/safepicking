#!/usr/bin/env python

import argparse
import datetime
import json
import logging
import socket

from loguru import logger
import numpy as np
import path
import pytz
import torch

from yarr.replay_buffer.prioritized_replay_buffer import (
    PrioritizedReplayBuffer,  # NOQA
)
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import (
    PyTorchReplayBuffer,  # NOQA
)
from yarr.runners.env_runner import EnvRunner
from yarr.runners.pytorch_train_runner import PyTorchTrainRunner

import mercury

from _agent import DqnAgent
from _env import PickFromPileEnv
from _rollout_generator import RolloutGenerator
from _stat_accumulator import SimpleAccumulator


here = path.Path(__file__).abspath().parent


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", required=True, help="name")
    parser.add_argument(
        "--model",
        choices=[
            "closedloop_pose_net",
            "openloop_pose_net",
            "conv_net",
            "semantic_conv_net",
            "fusion_net",
        ],
        help="model",
        required=True,
    )
    parser.add_argument(
        "--train-envs", type=int, default=5, help="number of train envs"
    )
    parser.add_argument(
        "--eval-envs", type=int, default=1, help="number of eval envs"
    )
    parser.add_argument(
        "--epsilon-max-step", type=int, default=5000, help="epsilon max step"
    )
    parser.add_argument("--device", default="cuda:0", help="device")
    parser.add_argument(
        "--use-reward-translation",
        type=int,
        default=1,
        choices=[0, 1],
        help="use reward translation",
    )
    parser.add_argument(
        "--use-reward-max-velocity",
        type=int,
        default=0,
        choices=[0, 1],
        help="use reward max velocity",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        help="gamma",
    )
    parser.add_argument(
        "--episode-length", type=int, default=5, help="episode length"
    )
    parser.add_argument(
        "--action-frame",
        choices=["object", "ee"],
        default="object",
        help="action frame",
    )
    args = parser.parse_args()

    hparams = args.__dict__.copy()

    now = datetime.datetime.now(pytz.timezone("Japan"))
    hparams["timestamp"] = now.isoformat()

    log_dir = now.strftime("%Y%m%d_%H%M%S") + "-" + hparams["name"]
    log_dir = here / "logs" / log_dir
    log_dir.makedirs_p()

    hparams["git_hash"] = mercury.utils.git_hash(cwd=here, log_dir=log_dir)
    hparams["hostname"] = socket.gethostname()

    with open(log_dir / "hparams.json", "w") as f:
        json.dump(hparams, f, indent=2)

    if hparams["device"] != "cpu":
        assert torch.cuda.is_available()
        assert torch.backends.cudnn.enabled

    logger.remove()
    logger.add(log_dir / "env.log", level="INFO")
    logging.basicConfig(
        filename=log_dir / "runner.log",
        format="%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)s - %(message)s",  # NOQA
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    # Setup env
    ###################

    env = PickFromPileEnv(
        gui=False,
        use_reward_translation=hparams["use_reward_translation"],
        use_reward_max_velocity=hparams["use_reward_max_velocity"],
        episode_length=hparams["episode_length"],
        action_frame=hparams["action_frame"],
    )

    # Setup replay buffer
    #####################

    replay_buffer = PrioritizedReplayBuffer(
        batch_size=128,
        timesteps=1,
        replay_capacity=50000,
        gamma=hparams["gamma"],
        action_shape=env.action_shape,
        action_dtype=np.int32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=env.observation_elements,
        save_dir=None,
    )

    wrapped_replay = PyTorchReplayBuffer(replay_buffer)

    # Setup rl algorithm
    ####################

    agent = DqnAgent(
        epsilon_max_step=hparams["epsilon_max_step"],
        gamma=hparams["gamma"],
        env=env,
        model=hparams["model"],
    )

    stat_accumulator = SimpleAccumulator(maxlen=100)

    env_runner = EnvRunner(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        train_envs=hparams["train_envs"],
        eval_envs=hparams["eval_envs"],
        episodes=999999,
        episode_length=env.episode_length,
        stat_accumulator=stat_accumulator,
        weightsdir=log_dir / "weights",
        max_fails=0,
        rollout_generator=RolloutGenerator(),
    )

    train_runner = PyTorchTrainRunner(
        agent=agent,
        env_runner=env_runner,
        wrapped_replay_buffer=[wrapped_replay],
        train_device=hparams["device"],
        stat_accumulator=stat_accumulator,
        iterations=100000,
        logdir=log_dir,
        log_freq=10,
        transitions_before_train=1000,
        weightsdir=log_dir / "weights",
        save_freq=100,
        replay_ratio=16,
    )

    train_runner.start()


if __name__ == "__main__":
    main()
