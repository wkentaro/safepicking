import torch


class ConvNet(torch.nn.Module):
    def __init__(self, episode_length, semantic=False):
        super().__init__()

        self._semantic = semantic

        # heightmap: 1
        # maskmap: 1
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1 + 1, 4, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )

        # h: 32
        # actions: 6
        # ee_poses: episode_length * 7
        # object_label: 7
        # object_pose: 7
        in_channels = 4 + 8 + 16 + 32 + 64 + 6 + episode_length * 7
        if self._semantic:
            in_channels += 7 + 7
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
        object_label=None,
        object_pose=None,
    ):
        B = heightmap.shape[0]
        A = actions.shape[0]
        H, W = heightmap.shape[1:]

        h = torch.cat(
            [heightmap[:, None, :, :], maskmap[:, None, :, :].float()], dim=1
        )
        h_conv1 = self.conv1(h)
        h_conv2 = self.conv2(h_conv1)
        h_conv3 = self.conv3(h_conv2)
        h_conv4 = self.conv4(h_conv3)
        h_conv5 = self.conv5(h_conv4)

        yx = torch.tensor(h_conv1.shape[2:]) // 2
        h_conv1 = h_conv1[:, :, yx[0], yx[1]]
        yx = torch.tensor(h_conv2.shape[2:]) // 2
        h_conv2 = h_conv2[:, :, yx[0], yx[1]]
        yx = torch.tensor(h_conv3.shape[2:]) // 2
        h_conv3 = h_conv3[:, :, yx[0], yx[1]]
        yx = torch.tensor(h_conv4.shape[2:]) // 2
        h_conv4 = h_conv4[:, :, yx[0], yx[1]]
        yx = torch.tensor(h_conv5.shape[2:]) // 2
        h_conv5 = h_conv5[:, :, yx[0], yx[1]]

        h = torch.cat([h_conv1, h_conv2, h_conv3, h_conv4, h_conv5], dim=1)

        h = h[:, None, :].repeat_interleave(A, dim=1)
        h_action = actions[None, :, :].repeat_interleave(B, dim=0)
        h_ee_pose = ee_poses[:, None, :, :].repeat_interleave(A, dim=1)
        h_ee_pose = h_ee_pose.reshape(B, A, -1)

        if self._semantic:
            h_semantic = torch.cat([object_label, object_pose], dim=0)
            h_semantic = h_semantic[None, :].repeat_interleave(B, dim=0)
            h_semantic = h_semantic[:, None, :].repeat_interleave(A, dim=1)
            h = torch.cat([h, h_action, h_ee_pose, h_semantic], dim=2)
        else:
            h = torch.cat([h, h_action, h_ee_pose], dim=2)

        h = h.reshape(B * A, -1)

        h = self.mlp(h)

        h = h.reshape(B, A, 2)

        return h
