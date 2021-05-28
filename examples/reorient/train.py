#!/usr/bin/env python

import argparse
import collections
import datetime
import time

import numpy as np
import path
import pytz
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm


home = path.Path("~").expanduser()
here = path.Path(__file__).abspath().parent


class PoseEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # object_classes: 22, object_poses: 7
        self.fc_encoder = torch.nn.Sequential(
            torch.nn.Linear(22 + 7, 32),
            torch.nn.ReLU(),
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=32,
                nhead=4,
                dim_feedforward=64,
            ),
            num_layers=2,
            norm=None,
        )

    def forward(self, object_classes, object_poses):
        h = torch.cat([object_classes, object_poses], dim=2)

        B, O, _ = h.shape
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

        # object_poses: (O, 7)
        self.encoder_object_poses = PoseEncoder()

        # reorient_pose: 7
        self.encoder_reorient_pose = torch.nn.Sequential(
            torch.nn.Linear(7, 32),
            torch.nn.ReLU(),
        )
        self.encoder_object_poses_and_reorient_pose = torch.nn.Sequential(
            torch.nn.Linear(32 + 32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
        )
        self.fc_auc = torch.nn.Sequential(
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid(),
        )

        # object_class: 22, grasp_pose: 7, initial_pose: 7, reorient_pose: 7
        self.encoder_action = torch.nn.Sequential(
            torch.nn.Linear(22 + 7 + 7 + 7, 32),
            torch.nn.ReLU(),
        )
        self.encoder_object_poses_and_action = torch.nn.Sequential(
            torch.nn.Linear(32 + 32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
        )
        self.fc_success = torch.nn.Sequential(
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid(),
        )
        self.fc_length = torch.nn.Sequential(
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid(),
        )

    def forward(
        self,
        object_classes,
        object_poses,
        object_class,
        grasp_pose,
        initial_pose,
        reorient_pose,
    ):
        h_object_poses = self.encoder_object_poses(
            object_classes, object_poses
        )

        # fc_auc
        h_reorient_pose = self.encoder_reorient_pose(reorient_pose)
        h_object_poses_and_reorient_pose = torch.cat(
            [h_object_poses, h_reorient_pose], dim=1
        )
        h_object_poses_and_reorient_pose = (
            self.encoder_object_poses_and_reorient_pose(
                h_object_poses_and_reorient_pose
            )
        )
        fc_auc = self.fc_auc(h_object_poses_and_reorient_pose)[:, 0]

        # fc_success, fc_length
        h_action = torch.cat(
            [object_class, grasp_pose, initial_pose, reorient_pose], dim=1
        )
        h_action = self.encoder_action(h_action)

        h_object_poses_and_action = torch.cat(
            [h_object_poses, h_action], dim=1
        )
        h_object_poses_and_action = self.encoder_object_poses_and_action(
            h_object_poses_and_action
        )

        fc_success = self.fc_success(h_object_poses_and_action)[:, 0]
        fc_length = self.fc_length(h_object_poses_and_action)[:, 0]
        return fc_success, fc_length, fc_auc


class Dataset(torch.utils.data.Dataset):

    ROOT_DIR = home / "data/mercury/reorient/n_class_5"

    TRAIN_SIZE = 20000
    EVAL_SIZE = 5000

    JS_PLACE_LENGTH_SCALING = 12

    def __init__(self, split):
        self._split = split

    def __len__(self):
        if self._split == "train":
            return self.TRAIN_SIZE
        elif self._split == "val":
            return self.EVAL_SIZE
        else:
            raise ValueError

    def __getitem__(self, i):
        if self._split == "val":
            i += self.TRAIN_SIZE
        data = np.load(self.ROOT_DIR / f"{i:08d}.npz")
        success = ~np.isnan(data["js_place_length"])
        if success:
            assert (
                0 <= data["js_place_length"] <= self.JS_PLACE_LENGTH_SCALING
            ), data["js_place_length"]
        object_classes = np.eye(22)[data["object_classes"]]
        object_class = object_classes[data["object_fg_flags"]][0]
        return dict(
            object_classes=object_classes,
            object_poses=data["object_poses"],
            object_class=object_class,
            grasp_pose=data["grasp_pose"],
            initial_pose=data["initial_pose"],
            reorient_pose=data["reorient_pose"],
            js_place_length=data["js_place_length"]
            / self.JS_PLACE_LENGTH_SCALING,
            auc=data["auc"],
            success=success,
        )


def epoch_loop(
    epoch, is_training, model, data_loader, summary_writer, optimizer=None
):
    if is_training:
        assert optimizer is not None

    train_or_val = "train" if is_training else "val"

    model.train(is_training)

    losses = collections.defaultdict(list)
    for iteration, batch in enumerate(
        tqdm.tqdm(
            data_loader,
            desc=f"{train_or_val.capitalize()} loop",
            leave=False,
            ncols=100,
        )
    ):
        success_pred, length_pred, auc_pred = model(
            object_classes=batch["object_classes"].float().cuda(),
            object_poses=batch["object_poses"].float().cuda(),
            object_class=batch["object_class"].float().cuda(),
            grasp_pose=batch["grasp_pose"].float().cuda(),
            initial_pose=batch["initial_pose"].float().cuda(),
            reorient_pose=batch["reorient_pose"].float().cuda(),
        )
        length_true = batch["js_place_length"].float().cuda()
        success_true = batch["success"].cuda()
        auc_true = batch["auc"].float().cuda()

        if is_training:
            optimizer.zero_grad()

        loss_auc = torch.nn.functional.smooth_l1_loss(auc_pred, auc_true)
        loss_success = torch.nn.functional.binary_cross_entropy(
            success_pred, success_true.float()
        )
        loss_length = torch.nn.functional.smooth_l1_loss(
            length_pred[success_true], length_true[success_true]
        )
        loss = loss_success + loss_length + loss_auc

        losses["loss_auc"].append(loss_auc.item())
        losses["loss_success"].append(loss_success.item())
        losses["loss_length"].append(loss_length.item())
        losses["loss"].append(loss.item())

        if is_training:
            loss.backward()
            optimizer.step()

            for key, values in losses.items():
                summary_writer.add_scalar(
                    f"train/{key}",
                    values[-1],
                    global_step=len(data_loader) * epoch + iteration,
                    walltime=time.time(),
                )

    return losses


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
        here / "logs" / now.strftime("%Y%m%d_%H%M%S.%f") + "-" + args.name
    )
    log_dir.makedirs_p()

    writer = SummaryWriter(log_dir=log_dir)

    eval_loss_init = None
    eval_loss_min_epoch = -1
    eval_loss_min = np.inf
    with tqdm.trange(1000, ncols=100) as pbar:
        for epoch in pbar:
            pbar.set_description(
                "Epoch loop "
                f"(eval_loss_min={eval_loss_min:.3f} @{eval_loss_min_epoch})"
            )
            pbar.refresh()

            epoch_loop(
                epoch=epoch,
                is_training=True,
                model=model,
                data_loader=loader_train,
                summary_writer=writer,
                optimizer=optimizer,
            )

            with torch.no_grad():
                losses = epoch_loop(
                    epoch=epoch,
                    is_training=False,
                    model=model,
                    data_loader=loader_val,
                    summary_writer=writer,
                    optimizer=optimizer,
                )

            for key, values in losses.items():
                writer.add_scalar(
                    f"val/{key}",
                    np.mean(values),
                    global_step=len(loader_train) * (epoch + 1),
                    walltime=time.time(),
                )

            eval_loss = np.mean(losses["loss"])
            if epoch == 0:
                eval_loss_init = eval_loss
            if eval_loss < eval_loss_min:
                model_file = (
                    log_dir / f"models/model_best-epoch_{epoch:04d}.pth"
                )
                model_file.parent.makedirs_p()
                torch.save(model.state_dict(), model_file)
                eval_loss_min_epoch = epoch
                eval_loss_min = eval_loss

            if eval_loss > eval_loss_init:
                break


if __name__ == "__main__":
    main()
