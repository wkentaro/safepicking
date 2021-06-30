#!/usr/bin/env python

import argparse
import datetime
import json
import pickle
import time

import numpy as np
import path
import pytz
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm

import mercury


home = path.Path("~").expanduser()
here = path.Path(__file__).abspath().parent


class PoseEncoder(torch.nn.Module):
    def __init__(self, out_channels, nhead, num_layers):
        super().__init__()
        # object_fg_flags: 1
        # object_labels: 7
        # object_poses: 7
        # grasp_pose: 6
        # reorient_pose: 7
        self.fc_encoder = torch.nn.Sequential(
            torch.nn.Linear(1 + 7 + 7 + 6 + 7, out_channels),
            torch.nn.ReLU(),
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=out_channels,
                nhead=nhead,
                dim_feedforward=out_channels * 2,
            ),
            num_layers=num_layers,
            norm=None,
        )

    def forward(
        self,
        object_fg_flags,
        object_labels,
        object_poses,
        grasp_pose,
        reorient_pose,
    ):
        B, O = object_fg_flags.shape

        object_fg_flags = object_fg_flags[:, :, None]
        grasp_pose = grasp_pose[:, None, :].repeat_interleave(O, dim=1)
        reorient_pose = reorient_pose[:, None, :].repeat_interleave(O, dim=1)

        h = torch.cat(
            [
                object_fg_flags,
                object_labels,
                object_poses,
                grasp_pose,
                reorient_pose,
            ],
            dim=2,
        )

        h = h.reshape(B * O, -1)
        h = self.fc_encoder(h)
        h = h.reshape(B, O, -1)

        h = h.permute(1, 0, 2)  # BOE -> OBE
        h = self.transformer_encoder(h)
        h = h.permute(1, 0, 2)  # OBE -> BOE

        h = h.mean(dim=1)

        return h


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_object_poses = PoseEncoder(32, nhead=4, num_layers=4)
        self.fc_pickable = torch.nn.Sequential(
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid(),
        )

    def forward(
        self,
        object_fg_flags,
        object_labels,
        object_poses,
        grasp_pose,
        reorient_pose,
    ):
        h = self.encoder_object_poses(
            object_fg_flags,
            object_labels,
            object_poses,
            grasp_pose,
            reorient_pose,
        )
        pickable = self.fc_pickable(h)[:, 0]

        return pickable


class Dataset(torch.utils.data.Dataset):

    ROOT_DIR = home / "data/mercury/reorient/pickable"

    def __init__(self, split):
        self._split = split

        self._files = {"train": [], "val": []}
        for i in range(0, 4500):
            seed_dir = self.ROOT_DIR / f"s-{i:08d}"
            if not seed_dir.exists():
                continue
            for pkl_file in sorted(seed_dir.walk("*.pkl")):
                self._files["train"].append(pkl_file)
        for i in range(4500, 5000):
            seed_dir = self.ROOT_DIR / f"s-{i:08d}"
            if not seed_dir.exists():
                continue
            for pkl_file in sorted(seed_dir.walk("*.pkl")):
                self._files["val"].append(pkl_file)

        for key, value in self._files.items():
            print(f"{key}: {len(value)}")

    def __len__(self):
        return len(self._files[self._split])

    def __getitem__(self, i):
        file = self._files[self._split][i]
        with open(file, "rb") as f:
            data = pickle.load(f)

        class_ids = [2, 3, 5, 11, 12, 15, 16]
        object_labels = []
        for object_class in data["object_classes"]:
            object_label = np.zeros(7)
            object_label[class_ids.index(object_class)] = 1
            object_labels.append(object_label)
        object_labels = np.array(object_labels, dtype=np.int32)

        object_fg_flags = data["object_fg_flags"]
        object_poses = data["object_poses"]

        ee_to_obj = np.hsplit(data["grasp_pose_wrt_obj"], [3])
        grasp_point_start = ee_to_obj[0]
        grasp_point_end = mercury.geometry.transform_points(
            [[0, 0, 1]], mercury.geometry.transformation_matrix(*ee_to_obj)
        )[0]
        grasp_pose = np.hstack([grasp_point_start, grasp_point_end])

        reorient_pose = data["reorient_pose"]
        pickable = data["pickable"]

        return dict(
            object_fg_flags=object_fg_flags,
            object_labels=object_labels,
            object_poses=object_poses,
            grasp_pose=grasp_pose,
            reorient_pose=reorient_pose,
            pickable=pickable,
        )


def epoch_loop(
    epoch, is_training, model, data_loader, summary_writer, optimizer=None
):
    if is_training:
        assert optimizer is not None

    train_or_val = "train" if is_training else "val"

    model.train(is_training)

    classes_pred = []
    classes_true = []
    losses = []
    for iteration, batch in enumerate(
        tqdm.tqdm(
            data_loader,
            desc=f"{train_or_val.capitalize()} loop",
            leave=False,
            ncols=100,
        )
    ):
        pickable_pred = model(
            object_fg_flags=batch["object_fg_flags"].float().cuda(),
            object_labels=batch["object_labels"].float().cuda(),
            object_poses=batch["object_poses"].float().cuda(),
            grasp_pose=batch["grasp_pose"].float().cuda(),
            reorient_pose=batch["reorient_pose"].float().cuda(),
        )
        pickable_true = batch["pickable"].float().cuda()

        if is_training:
            optimizer.zero_grad()

        classes_true.extend((pickable_true > 0.5).cpu().numpy().tolist())
        classes_pred.extend((pickable_pred > 0.5).cpu().numpy().tolist())

        loss = torch.nn.functional.binary_cross_entropy(
            pickable_pred, pickable_true
        )
        losses.append(loss.item())

        if is_training:
            loss.backward()
            optimizer.step()

            summary_writer.add_scalar(
                "train/loss",
                loss.item(),
                global_step=len(data_loader) * epoch + iteration,
                walltime=time.time(),
            )
    classes_pred = np.array(classes_pred)
    classes_true = np.array(classes_true)

    metrics = dict()
    if not is_training:
        metrics["loss"] = np.mean(losses)
        tp = (classes_true & classes_pred).sum()
        fp = (~classes_true & classes_pred).sum()
        tn = (~classes_true & ~classes_pred).sum()
        fn = (classes_true & ~classes_pred).sum()
        metrics["accuracy"] = (tp + tn) / (tp + fp + tn + fn)
        metrics["precision"] = tp / (tp + fp)
        metrics["recall"] = tp / (tp + fn)
        metrics["specificity"] = tn / (tn + fp)
        metrics["balanced"] = (metrics["recall"] + metrics["specificity"]) / 2
        metrics["f1"] = 2 / (1 / metrics["precision"] + 1 / metrics["recall"])
        for key, value in metrics.items():
            summary_writer.add_scalar(
                f"val/{key}",
                value,
                global_step=len(data_loader) * epoch + iteration,
                walltime=time.time(),
            )

    return metrics


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", required=True, help="name")
    args = parser.parse_args()

    data_train = Dataset(split="train")
    loader_train = torch.utils.data.DataLoader(
        data_train, batch_size=256, shuffle=True, drop_last=True
    )
    data_val = Dataset(split="val")
    loader_val = torch.utils.data.DataLoader(
        data_val, batch_size=256, shuffle=False
    )

    model = Model()
    model.cuda()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    now = datetime.datetime.now(pytz.timezone("Asia/Tokyo"))
    log_dir = (
        here / "logs/pickable" / now.strftime("%Y%m%d_%H%M%S.%f")
        + "-"
        + args.name
    )
    log_dir.makedirs_p()

    git_hash = mercury.utils.git_hash(cwd=here, log_dir=log_dir)
    with open(log_dir / "params.json", "w") as f:
        json.dump({"git_hash": git_hash}, f)

    writer = SummaryWriter(log_dir=log_dir)

    max_metric = (-1, -np.inf)
    with tqdm.trange(-1, 200, ncols=100) as pbar:
        for epoch in pbar:
            pbar.set_description(
                f"Epoch loop ({max_metric[1]:.3f} @{max_metric[0]})"
            )
            pbar.refresh()

            # train epoch
            if epoch >= 0:
                epoch_loop(
                    epoch=epoch,
                    is_training=True,
                    model=model,
                    data_loader=loader_train,
                    summary_writer=writer,
                    optimizer=optimizer,
                )

            # val epoch
            with torch.no_grad():
                metrics = epoch_loop(
                    epoch=epoch,
                    is_training=False,
                    model=model,
                    data_loader=loader_val,
                    summary_writer=writer,
                    optimizer=optimizer,
                )
            if epoch >= 0 and metrics["f1"] > max_metric[1]:
                model_file = (
                    log_dir / f"models/model_best-epoch_{epoch:04d}.pt"
                )
                model_file.parent.makedirs_p()
                torch.save(model.state_dict(), model_file)
                max_metric = (epoch, metrics["f1"])


if __name__ == "__main__":
    main()
