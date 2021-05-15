#!/usr/bin/env python

import collections

import numpy as np
import path
import torch
import tqdm


home = path.Path("~").expanduser()
here = path.Path(__file__).abspath().parent


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            collections.OrderedDict(
                fc1=torch.nn.Linear(7 + 7, 32),
                relu1=torch.nn.ReLU(),
                fc2=torch.nn.Linear(32, 32),
                relu2=torch.nn.ReLU(),
                fc3=torch.nn.Linear(32, 32),
                relu3=torch.nn.ReLU(),
            )
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

    def forward(self, grasp_pose, reorient_pose):
        h = torch.cat([grasp_pose, reorient_pose], dim=1)
        h = self.features(h)
        fc_success = self.fc_success(h)[:, 0]
        fc_length = self.fc_length(h)[:, 0]
        fc_auc = self.fc_auc(h)[:, 0]
        return fc_success, fc_length, fc_auc


class Dataset(torch.utils.data.Dataset):

    ROOT_DIR = home / "data/mercury/reorient/00001000"

    TRAIN_SIZE = 8000
    EVAL_SIZE = 2000

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
        data = np.load(self.ROOT_DIR / f"0005_{i:04d}.npz")
        success = ~np.isnan(data["js_place_length"])
        if success:
            assert (
                0 <= data["js_place_length"] <= self.JS_PLACE_LENGTH_SCALING
            ), data["js_place_length"]
        return dict(
            grasp_pose=data["grasp_pose"],
            reorient_pose=data["reorient_pose"],
            js_place_length=data["js_place_length"]
            / self.JS_PLACE_LENGTH_SCALING,
            auc=data["auc"],
            success=success,
        )


def main():
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

    log_dir = here / "logs/test"
    log_dir.makedirs_p()

    eval_loss_min = np.inf
    with tqdm.trange(1000) as pbar:
        for epoch in pbar:
            pbar.set_description(
                f"Epoch loop (eval_loss_min={eval_loss_min:.3f})"
            )
            pbar.refresh()

            model.train()

            for batch in tqdm.tqdm(
                loader_train, desc="Train loop", leave=False
            ):
                success_pred, length_pred, auc_pred = model(
                    grasp_pose=batch["grasp_pose"].float().cuda(),
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

            model.eval()

            losses = []
            with torch.no_grad():
                for batch in tqdm.tqdm(
                    loader_val, desc="Val loop", leave=False
                ):
                    success_pred, length_pred, auc_pred = model(
                        grasp_pose=batch["grasp_pose"].float().cuda(),
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

            if eval_loss < eval_loss_min:
                torch.save(
                    model.state_dict(),
                    log_dir / f"model_best-epoch_{epoch:04d}.pth",
                )
                eval_loss_min = eval_loss


if __name__ == "__main__":
    main()
