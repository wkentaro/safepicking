#!/usr/bin/env python

import argparse
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

        self.pose_encoder = PoseEncoder()

        # grasp_pose: 7, initial_pose: 7, reorient_pose: 7
        self.action_encoder = torch.nn.Sequential(
            torch.nn.Linear(7 + 7 + 7, 32),
            torch.nn.ReLU(),
        )

        self.encoder = torch.nn.Sequential(
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
        self.fc_auc = torch.nn.Sequential(
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid(),
        )

    def forward(
        self,
        object_classes,
        object_poses,
        grasp_pose,
        initial_pose,
        reorient_pose,
    ):
        h_pose = self.pose_encoder(object_classes, object_poses)

        h_action = torch.cat([grasp_pose, initial_pose, reorient_pose], dim=1)
        h_action = self.action_encoder(h_action)

        h = torch.cat([h_pose, h_action], dim=1)
        h = self.encoder(h)

        fc_success = self.fc_success(h)[:, 0]
        fc_length = self.fc_length(h)[:, 0]
        fc_auc = self.fc_auc(h)[:, 0]
        return fc_success, fc_length, fc_auc


class Dataset(torch.utils.data.Dataset):

    ROOT_DIR = home / "data/mercury/reorient/train"

    TRAIN_SIZE = 40000
    EVAL_SIZE = 7777

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
        return dict(
            object_classes=object_classes,
            object_poses=data["object_poses"],
            grasp_pose=data["grasp_pose"],
            initial_pose=data["initial_pose"],
            reorient_pose=data["reorient_pose"],
            js_place_length=data["js_place_length"]
            / self.JS_PLACE_LENGTH_SCALING,
            auc=data["auc"],
            success=success,
        )


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

    eval_loss_min = np.inf
    with tqdm.trange(1000, ncols=100) as pbar:
        for epoch in pbar:
            pbar.set_description(
                f"Epoch loop (eval_loss_min={eval_loss_min:.3f})"
            )
            pbar.refresh()

            model.train()

            for iteration, batch in enumerate(
                tqdm.tqdm(
                    loader_train, desc="Train loop", leave=False, ncols=100
                )
            ):
                success_pred, length_pred, auc_pred = model(
                    object_classes=batch["object_classes"].float().cuda(),
                    object_poses=batch["object_poses"].float().cuda(),
                    grasp_pose=batch["grasp_pose"].float().cuda(),
                    initial_pose=batch["initial_pose"].float().cuda(),
                    reorient_pose=batch["reorient_pose"].float().cuda(),
                )
                length_true = batch["js_place_length"].float().cuda()
                success_true = batch["success"].cuda()
                auc_true = batch["auc"].float().cuda()

                optimizer.zero_grad()

                loss_success = torch.nn.functional.binary_cross_entropy(
                    success_pred, success_true.float()
                )
                loss_length = torch.nn.functional.smooth_l1_loss(
                    length_pred[success_true], length_true[success_true]
                )
                loss_auc = torch.nn.functional.smooth_l1_loss(
                    auc_pred[success_true], auc_true[success_true]
                )
                loss = loss_success + loss_length + loss_auc
                loss.backward()

                optimizer.step()

                writer.add_scalar(
                    "train/loss",
                    loss.item(),
                    global_step=len(loader_train) * epoch + iteration,
                    walltime=time.time(),
                )

            model.eval()

            losses = []
            with torch.no_grad():
                for batch in tqdm.tqdm(
                    loader_val, desc="Val loop", leave=False, ncols=100
                ):
                    success_pred, length_pred, auc_pred = model(
                        object_classes=batch["object_classes"].float().cuda(),
                        object_poses=batch["object_poses"].float().cuda(),
                        grasp_pose=batch["grasp_pose"].float().cuda(),
                        initial_pose=batch["initial_pose"].float().cuda(),
                        reorient_pose=batch["reorient_pose"].float().cuda(),
                    )
                    length_true = batch["js_place_length"].float().cuda()
                    success_true = batch["success"].cuda()
                    auc_true = batch["auc"].float().cuda()

                    loss_success = torch.nn.functional.binary_cross_entropy(
                        success_pred, success_true.float()
                    )
                    loss_length = torch.nn.functional.smooth_l1_loss(
                        length_pred[success_true], length_true[success_true]
                    )
                    loss_auc = torch.nn.functional.smooth_l1_loss(
                        auc_pred[success_true], auc_true[success_true]
                    )
                    loss = loss_success + loss_length + loss_auc

                    losses.append(loss.item())
            eval_loss = np.mean(losses)

            writer.add_scalar(
                "val/loss",
                eval_loss,
                global_step=len(loader_train) * epoch + iteration,
                walltime=time.time(),
            )

            if eval_loss < eval_loss_min:
                model_file = (
                    log_dir / f"models/model_best-epoch_{epoch:04d}.pth"
                )
                model_file.parent.makedirs_p()
                torch.save(model.state_dict(), model_file)
                eval_loss_min = eval_loss


if __name__ == "__main__":
    main()
