import torch


class FusionNet(torch.nn.Module):
    def __init__(self, episode_length):
        super().__init__()

        # heightmap: 1
        # maskmap: 1
        self.conv_encoder_raw = torch.nn.Sequential(
            torch.nn.Conv2d(1 + 1, 4, kernel_size=3, stride=1, padding=1),
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

        self.fc_encoder_raw = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
        )

        # object_labels: 7
        # object_poses: 7
        # grasp_flags: 1
        # actions: 6
        # ee_poses: episode_length * 7
        in_channels = 7 + 7 + 1 + 6 + episode_length * 7
        self.fc_encoder_pose = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 32),
            torch.nn.ReLU(),
        )

        self.fc_encoder_mix = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
        )

        self.transformer_mix = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=32,
                nhead=4,
                dim_feedforward=64,
            ),
            num_layers=4,
            norm=None,
        )

        self.fc_output = torch.nn.Sequential(
            torch.nn.Linear(32, 2),
        )

    def forward(
        self,
        heightmap,
        maskmap,
        object_labels,
        object_poses,
        grasp_flags,
        ee_poses,
        actions,
    ):
        B = heightmap.shape[0]
        A = actions.shape[0]

        h_raw = torch.cat(
            [heightmap[:, None, :, :], maskmap[:, None, :, :].float()], dim=1
        )
        h_raw = self.conv_encoder_raw(h_raw)
        h_raw = h_raw.reshape(B, -1)
        h_raw = self.fc_encoder_raw(h_raw)

        # -----------------------------------------------------------------------------

        B, O = grasp_flags.shape
        A, _ = actions.shape

        ee_poses = ee_poses.reshape(B, -1)[:, None, :].repeat(1, O, 1)
        h_pose = torch.cat(
            [object_labels, object_poses, grasp_flags[:, :, None], ee_poses],
            dim=2,
        )

        # B, A, O, C
        h_pose = h_pose[:, None, :, :].repeat(1, A, 1, 1)
        h_action = actions[None, :, None, :].repeat(B, 1, O, 1)

        h_pose = torch.cat([h_pose, h_action], dim=3)

        h_pose = h_pose.reshape(B * A * O, h_pose.shape[-1])  # B*A*O, C
        h_pose = self.fc_encoder_pose(h_pose)
        h_pose = h_pose.reshape(B * A, O, h_pose.shape[-1])  # B*A, O, C

        h_raw = h_raw[:, None, None, :].repeat(1, A, O, 1)
        h_raw = h_raw.reshape(B * A, O, h_raw.shape[-1])

        h = torch.cat([h_raw, h_pose], dim=2)
        h = self.fc_encoder_mix(h)

        h = h.permute(1, 0, 2)  # B*A, O, C -> O, B*A, C
        h = self.transformer_mix(h)
        h = h.permute(1, 0, 2)  # O, B*A, C -> B*A, O, C

        h = h.mean(dim=1)  # B*A, O, C -> B*A, C

        h = self.fc_output(h)  # B*A, 1

        h = h.reshape(B, A, 2)

        return h
