import torch


class PoseNet(torch.nn.Module):
    def __init__(self, episode_length, openloop):
        super().__init__()

        self._openloop = openloop

        # ee_pose: 7
        # object_labels: 7
        # object_poses: 7
        # grasp_flags: 1
        # actions: 6
        in_channels = 7 + 7 + 7 + 1 + 6
        if self._openloop:
            in_channels += (episode_length - 1) * 7
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
        ee_pose,
        object_labels,
        object_poses,
        grasp_flags,
        actions,
        **kwargs,
    ):
        B, O = grasp_flags.shape
        A, _ = actions.shape

        h_ee_pose = ee_pose
        h_object = torch.cat(
            [object_labels, object_poses, grasp_flags[:, :, None]], dim=2
        )
        if self._openloop:
            past_grasped_object_poses = kwargs["past_grasped_object_poses"]
            past_grasped_object_poses = past_grasped_object_poses.reshape(
                B, -1
            )[:, None, :].repeat(1, O, 1)
            h_object = torch.cat([h_object, past_grasped_object_poses], dim=2)
        h_action = actions

        # B, A, O, C
        h_ee_pose = h_ee_pose[:, None, None, :].repeat(1, A, O, 1)
        h_object = h_object[:, None, :, :].repeat(1, A, 1, 1)
        h_action = h_action[None, :, None, :].repeat(B, 1, O, 1)

        h = torch.cat([h_ee_pose, h_object, h_action], dim=3)

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
