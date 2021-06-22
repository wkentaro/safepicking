import torch


class PoseNet(torch.nn.Module):
    def __init__(self, episode_length, openloop):
        super().__init__()

        # object_labels: 7
        # object_poses: 7
        # grasp_flags: 1
        # actions: 6
        # ee_poses: episode_length * 7 or 7
        if openloop:
            in_channels = 7 + 7 + 1 + 6 + episode_length * 7
        else:
            in_channels = 7 + 7 + 1 + 6 + 7
        self.fc_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 32),
            torch.nn.ReLU(),
        )
        self.transformer_object = torch.nn.TransformerEncoder(
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
        object_labels,
        object_poses,
        grasp_flags,
        ee_poses,
        actions,
    ):
        B, O = grasp_flags.shape
        A, _ = actions.shape

        h = torch.cat(
            [object_labels, object_poses, grasp_flags[:, :, None]], dim=2
        )
        ee_poses = ee_poses.reshape(B, -1)[:, None, :].repeat(1, O, 1)
        h = torch.cat([h, ee_poses], dim=2)
        h_action = actions

        # B, A, O, C
        h = h[:, None, :, :].repeat(1, A, 1, 1)
        h_action = h_action[None, :, None, :].repeat(B, 1, O, 1)

        h = torch.cat([h, h_action], dim=3)

        h = h.reshape(B * A * O, h.shape[-1])  # B*A*O, C
        h = self.fc_encoder(h)
        h = h.reshape(B * A, O, h.shape[-1])  # B*A, O, C

        h = h.permute(1, 0, 2)  # B*A, O, C -> O, B*A, C
        h = self.transformer_object(h)
        h = h.permute(1, 0, 2)  # O, B*A, C -> B*A, O, C

        h = h.mean(dim=1)  # B*A, O, C -> B*A, C

        h = self.fc_output(h)  # B*A, 1

        h = h.reshape(B, A, 2)

        return h
