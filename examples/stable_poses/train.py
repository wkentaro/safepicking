#!/usr/bin/env python

import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm


class Unet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        C = 8
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, C, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(C, C * 2, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(C * 2, C * 4, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(C * 4, C * 8, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(C * 8 + 3, C * 8),
            torch.nn.ReLU(),
            torch.nn.Linear(C * 8, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, rgb, yx, euler):
        conv1 = self.conv1(rgb)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)

        h = F.interpolate(
            conv4, size=rgb.shape[-2:], mode="bilinear", align_corners=True
        )

        y = yx[:, :, 0]
        x = yx[:, :, 1]

        B = y.shape[0]
        assert B == 1

        h = h[:, :, y[0], x[0]]
        h = h.permute(0, 2, 1)
        h = torch.cat([h, euler], axis=2)

        return self.fc(h)[:, :, 0]


class Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1001

    def __getitem__(self, i):
        data = dict(
            pickle.load(
                open(
                    f"/home/wkentaro/data/mercury/stable_poses/{i:08d}.pkl",
                    "rb",
                )
            )
        )

        rgb = (data["rgb"] / 255).astype(np.float32).transpose(2, 0, 1)

        yx = np.array(data["yx"]).astype(np.int64)
        euler = np.array(data["euler"]).astype(np.float32)
        auc = np.array(data["auc"]).astype(np.float32)

        p = np.random.permutation(len(yx))[:1000]

        return dict(
            rgb=rgb,
            yx=yx[p],
            euler=euler[p],
            stable=auc[p],
        )


def main():
    dataset = Dataset()
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True
    )

    model = Unet()
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batches = list(iter(data_loader))
    loss_min_epoch = -1
    loss_min = np.inf
    with tqdm.trange(1000) as pbar:
        for epoch in pbar:
            pbar.set_description(f"loss_min={loss_min:.4f} @{loss_min_epoch}")
            pbar.refresh()

            losses = []
            for batch in tqdm.tqdm(batches, leave=False):
                stable_pred = model(
                    batch["rgb"].cuda(),
                    batch["yx"].cuda(),
                    batch["euler"].cuda(),
                )
                stable_true = batch["stable"].cuda().float()

                loss = F.smooth_l1_loss(stable_pred, stable_true, beta=0.01)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            loss_mean = np.mean(losses)
            if loss_mean < loss_min:
                torch.save(model.state_dict(), "model.pth")
                loss_min = loss_mean
                loss_min_epoch = epoch


if __name__ == "__main__":
    main()
