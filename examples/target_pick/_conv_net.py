import torch


class ConvNet(torch.nn.Module):
    def __init__(
        self,
        episode_length,
        position=False,
        pose=False,
    ):
        super().__init__()

        self._position = position
        self._pose = pose

        # heightmap: 1
        # maskmap: 1
        in_channels = 1 + 1
        if self._position:
            # positionmap: 3
            in_channels += 3
        if self._pose:
            # posemap: 3
            in_channels += 3
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, 4, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(4, stride=4),
        )

        # h: 64
        # actions: 6
        # ee_poses: episode_length * 7
        in_channels = 64 + 6 + episode_length * 7
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
        )

    def forward(
        self,
        heightmap,
        maskmap,
        ee_poses,
        actions,
        positionmap=None,
        posemap=None,
    ):
        B = heightmap.shape[0]
        A = actions.shape[0]
        H, W = heightmap.shape[1:]

        h = [heightmap[:, None, :, :], maskmap[:, None, :, :].float()]
        if self._position:
            h.append(positionmap.permute(0, 3, 1, 2))
        if self._pose:
            h.append(posemap.permute(0, 3, 1, 2))
        h = torch.cat(h, dim=1)
        h = self.encoder(h)
        h = h.reshape(B, -1)

        h = h[:, None, :].repeat_interleave(A, dim=1)
        h_action = actions[None, :, :].repeat_interleave(B, dim=0)
        h_ee_pose = ee_poses[:, None, :, :].repeat_interleave(A, dim=1)
        h_ee_pose = h_ee_pose.reshape(B, A, -1)
        h = torch.cat([h, h_action, h_ee_pose], dim=2)

        h = h.reshape(B * A, -1)

        h = self.mlp(h)

        h = h.reshape(B, A, 2)

        return h
