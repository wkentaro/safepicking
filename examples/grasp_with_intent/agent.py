import os
import queue

import imgviz
import numpy as np
import torch

from yarr.agents.agent import ActResult
from yarr.agents.agent import Agent
from yarr.agents.agent import ImageSummary
from yarr.agents.agent import ScalarSummary

from grasp_net import GraspNet


class DqnModel(torch.nn.Module):
    def __init__(self, env, model):
        super().__init__()
        del env

        self.module = GraspNet(model=model)

    def forward(self, observation):
        rgb = observation["rgb"].float() / 255
        depth = observation["depth"][:, None, :, :]
        ins = observation["ins"]

        return self.module(rgb=rgb, depth=depth, ins=ins)


class DqnAgent(Agent):
    def __init__(self, epsilon_max_step=5000, **kwargs):
        self._epsilon_max_step = epsilon_max_step
        self._kwargs = kwargs
        self._epsilon = np.nan
        self._losses = queue.deque(maxlen=18)
        self._act_summary = None

    def build(self, training, device=None):
        self.q = DqnModel(**self._kwargs).to(device).train(training)
        if training:
            self.q_target = DqnModel(**self._kwargs).to(device).train(False)
            for p in self.q_target.parameters():
                p.requires_grad = False
            self.optimizer = torch.optim.Adam(self.q.parameters(), lr=1e-3)
        else:
            for p in self.q.parameters():
                p.requires_grad = False

    def act(self, step, observation, deterministic, env):
        obs = {}
        for key in observation:
            obs[key] = torch.as_tensor(observation[key][:, 0])
            assert obs[key].shape[0] == 1

        with torch.no_grad():
            q = self.q(obs)
            q = q.numpy()
        assert q.shape[0] == 1
        assert q.shape[1] == 1

        self._act_summary = dict(
            step=step,
            observation=observation,
            deterministic=deterministic,
            q=q,
        )

        _, _, height, width = q.shape
        argsort = np.argsort(q[0, 0].flatten())[::-1]
        y = argsort // width
        x = argsort % width
        actions_select = np.stack((y, x), axis=1)

        if deterministic:
            for a in actions_select:
                act_result = ActResult(action=a)
                break
            else:
                act_result = ActResult(action=actions_select[0])
        else:
            self._epsilon = epsilon = self._get_epsilon(step)
            if np.random.random() < epsilon:
                for a in np.random.permutation(
                    np.argwhere(obs["fg_mask"][0].numpy())
                ):
                    act_result = ActResult(action=a)
                    break
            else:
                for a in actions_select:
                    act_result = ActResult(action=a)
                    break
                else:
                    act_result = ActResult(action=actions_select[0])
        return act_result

    def _get_epsilon(self, step):
        epsilon_init = 1
        epsilon_final = 0.01
        min_step = 0
        max_step = self._epsilon_max_step
        if max_step == min_step:
            alpha = 1
        else:
            alpha = min(1, max(0, (step - min_step) / (max_step - min_step)))
        epsilon = alpha * epsilon_final + (1 - alpha) * epsilon_init
        return epsilon

    def _get_sampling_weights(self, replay_sample):
        loss_weights = 1.0
        if "sampling_probabilities" in replay_sample:
            self.indicies = replay_sample["indices"]
            probs = replay_sample["sampling_probabilities"]
            loss_weights = 1.0 / torch.sqrt(probs + 1e-10)
            loss_weights /= torch.max(loss_weights)
        return loss_weights

    def update(self, step, replay_sample):
        update_dict = self._update_q(replay_sample)
        soft_updates(self.q, self.q_target, tau=0.005)
        return update_dict

    def _update_q(self, replay_sample):
        action = replay_sample["action"].long()
        reward = replay_sample["reward"]

        terminal = replay_sample["terminal"].float()

        def stack_timesteps(x):
            return torch.cat(torch.split(x, 1, dim=1), -1).squeeze(1)

        obs = {}
        obs_tp1 = {}
        for key in replay_sample:
            if key in [
                "action",
                "reward",
                "terminal",
                "timeout",
                "indices",
                "sampling_probabilities",
            ]:
                continue
            if key.endswith("_tp1"):
                obs_tp1[key[:-4]] = stack_timesteps(replay_sample[key])
            else:
                obs[key] = stack_timesteps(replay_sample[key])

        with torch.no_grad():
            qs_target = self.q_target(obs_tp1)
            assert qs_target.shape[1] == 1
            qs_target = qs_target.reshape(qs_target.shape[0], -1)
            q_target = torch.max(qs_target, dim=1).values
            q_target = reward + 0.99 * (1 - terminal) * q_target

        qs_pred = self.q(obs)
        assert qs_pred.shape[1] == 1

        q_pred = qs_pred[
            torch.arange(qs_pred.shape[0]), 0, action[:, 0], action[:, 1]
        ]

        self.optimizer.zero_grad()

        sampling_weigths = self._get_sampling_weights(replay_sample)
        q_delta = torch.nn.functional.smooth_l1_loss(
            q_pred, q_target, reduction="none"
        )
        q_loss = (q_delta * sampling_weigths).mean()

        bg_mask = 1 - obs["fg_mask"]
        q_bg_pred = []
        for b in range(bg_mask.shape[0]):
            y, x = torch.where(bg_mask[b])
            i = torch.randint(low=0, high=len(y), size=()).item()
            q_bg_pred.append(qs_pred[b, 0, y[i], x[i]])
        q_bg_pred = torch.stack(q_bg_pred)
        q_bg_delta = torch.nn.functional.smooth_l1_loss(
            q_bg_pred,
            torch.zeros_like(q_bg_pred),
            reduction="none",
        )
        q_bg_loss = (q_bg_delta * sampling_weigths).mean()

        loss = q_loss + q_bg_loss

        loss.backward()

        self.optimizer.step()

        self._losses.append(loss.item())
        priority = torch.sqrt(q_delta + 1e-10).detach()
        priority /= priority.max()
        prev_priority = replay_sample.get("priority", 0)

        return dict(priority=priority + prev_priority)

    def update_summaries(self):
        return [
            ScalarSummary("agent/loss", np.mean(self._losses)),
        ]

    def draw_act_summary(self):
        obs = self._act_summary["observation"]
        rgb = obs["rgb"][0, 0].transpose(1, 2, 0)
        ins = np.uint8(obs["ins"][0, 0].transpose(1, 2, 0) * 255)
        depth = obs["depth"][0, 0]
        q = self._act_summary["q"][0, 0]
        act_summary = imgviz.tile(
            [
                rgb,
                ins,
                imgviz.depth2rgb(depth),
                imgviz.depth2rgb(q, min_value=0, max_value=1),
            ],
            shape=(2, 2),
            border=(0, 0, 0),
        )
        return act_summary

    def act_summaries(self):
        summaries = [ScalarSummary("agent/epsilon", self._epsilon)]
        if self._act_summary:
            act_summary = self.draw_act_summary()
            summaries.append(
                ImageSummary(
                    "agent/act_summary", act_summary.transpose(2, 0, 1)
                )
            )
        return summaries

    def load_weights(self, save_dir):
        device = torch.device("cpu")
        self.q.load_state_dict(
            torch.load(os.path.join(save_dir, "q.pth"), map_location=device)
        )

    def save_weights(self, save_dir):
        torch.save(self.q.state_dict(), os.path.join(save_dir, "q.pth"))


def soft_updates(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )
