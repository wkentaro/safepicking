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
from yarr.utils.stat_accumulator import SimpleAccumulator

from agent import DqnAgent
import common_utils
from env import GraspWithIntentEnv
from rollout_generator import RolloutGenerator


here = path.Path(__file__).abspath().parent


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", required=True, help="name")
    parser.add_argument(
        "--model",
        required=True,
        choices=["rgb", "rgbd", "ins", "insd"],
        help="model",
    )
    parser.add_argument(
        "--train-envs", type=int, default=5, help="number of train envs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate",
    )
    parser.add_argument(
        "--epsilon-max-step", type=int, default=1000, help="epsilon max step"
    )
    parser.add_argument("--debug", action="store_true", help="debug")
    parser.add_argument("--device", default="cuda:0", help="device")
    args = parser.parse_args()

    hparams = args.__dict__.copy()

    now = datetime.datetime.now(pytz.timezone("Japan"))
    hparams["timestamp"] = now.isoformat()

    log_dir = now.strftime("%Y%m%d_%H%M%S") + "_" + hparams["name"]
    log_dir = here / "logs" / log_dir
    log_dir.makedirs_p()

    hparams["git_hash"] = common_utils.git_hash(cwd=here, log_dir=log_dir)
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

    env = GraspWithIntentEnv(gui=False)

    # Setup replay buffer
    #####################

    replay_buffer = PrioritizedReplayBuffer(
        batch_size=32,
        timesteps=1,
        replay_capacity=int(1e4),
        gamma=0.99,
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
        env=env,
        model=hparams["model"],
    )

    stat_accumulator = SimpleAccumulator()

    env_runner = EnvRunner(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        train_envs=hparams["train_envs"],
        eval_envs=1,
        episodes=999999,
        episode_length=1,
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
        transitions_before_train=320,
        weightsdir=log_dir / "weights",
        save_freq=10,
        max_replay_ratio=16,
    )

    train_runner.start()


if __name__ == "__main__":
    main()
