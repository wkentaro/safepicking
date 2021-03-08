#!/usr/bin/env python

import datetime
import numpy as np
import path
import torch
from torch.utils.tensorboard import SummaryWriter


class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        root_dir = path.Path("./data/disturbed_pile.v2")
        files = sorted(root_dir.listdir())
        assert len(files) == 1000

        if split == "train":
            self._files = files[:600]
        elif split == "val":
            self._files = files[600:800]
        else:
            assert split == "test"
            self._files = files[800:]

    def __len__(self):
        return len(self._files)

    def __getitem__(self, i):
        file = self._files[i]

        data = np.load(file)

        positions = data["positions"]
        quaternions = data["quaternions"]
        delta_positions = data["delta_positions"]

        poses = np.c_[positions, quaternions]
        poses = poses.reshape(-1).astype(np.float32)
        delta_positions = delta_positions.astype(np.float32)
        return poses, delta_positions


def get_model():
    channels = 64
    model = torch.nn.Sequential(
        torch.nn.Linear((3 + 4) * 8, channels),
        torch.nn.ReLU(),
        torch.nn.Linear(channels, channels),
        torch.nn.ReLU(),
        torch.nn.Linear(channels, channels),
        torch.nn.ReLU(),
        torch.nn.Linear(channels, channels),
        torch.nn.ReLU(),
        torch.nn.Linear(channels, 8),
        torch.nn.Sigmoid(),
    )
    model.to("cuda:0")
    return model


def main():
    data_train = Dataset(split="train")
    loader_train = torch.utils.data.DataLoader(
        data_train, batch_size=32, shuffle=True
    )
    data_val = Dataset(split="val")
    loader_val = torch.utils.data.DataLoader(data_val, batch_size=32)

    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    now = datetime.datetime.now()
    log_dir = path.Path("data/train") / now.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    loss_val_min = float("inf")
    for epoch in range(1000):
        losses = []
        for x, y_true in loader_train:
            x = x.to("cuda:0")
            y_true = y_true.to("cuda:0")

            y_pred = model(x)

            loss = torch.nn.functional.smooth_l1_loss(
                y_pred, y_true, reduction="none"
            )
            loss = loss.sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        loss_train = np.mean(losses)

        print(epoch, loss_train)

        writer.add_scalar("train/loss", loss_train, global_step=epoch)

        if epoch % 20 == 19:
            model.eval()

            losses = []
            for x, y_true in loader_val:
                x = x.to("cuda:0")
                y_true = y_true.to("cuda:0")

                with torch.no_grad():
                    y_pred = model(x)

                loss = torch.nn.functional.smooth_l1_loss(
                    y_pred, y_true, reduction="none"
                )
                loss = loss.sum(dim=1).mean()

                losses.append(loss.item())
            loss_val = np.mean(losses)

            writer.add_scalar("val/loss", loss_val, global_step=epoch)

            if loss_val < loss_val_min:
                print(epoch, "saving best model", loss_val)
                torch.save(model.state_dict(), log_dir / "model_best.pth")
                loss_val_min = loss_val

            model.train()


if __name__ == "__main__":
    main()
