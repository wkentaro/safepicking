import torch


class PoseNet(torch.nn.Module):
    def __init__(self, n_action):
        super().__init__()

        # object_labels: 7, object_poses: 7, grasp_flags: 1
        self.fc_object = torch.nn.Sequential(
            torch.nn.Linear(7 + 7 + 1, 32),
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

        self.fc_action = torch.nn.Sequential(
            torch.nn.Linear(6, 32),
            torch.nn.ReLU(),
        )

        self.fc_output = torch.nn.Sequential(
            torch.nn.Linear(64, 1),
        )

    def forward(
        self,
        object_labels,
        object_poses,
        grasp_flags,
        actions,
    ):
        B, O = grasp_flags.shape
        A, _ = actions.shape

        h_object = torch.cat(
            [object_labels, object_poses, grasp_flags[:, :, None]], dim=2
        )

        h_object = h_object.reshape(B * O, h_object.shape[2])
        h_object = self.fc_object(h_object)
        h_object = h_object.reshape(B, O, h_object.shape[1])  # BOE

        h_object = h_object.permute(1, 0, 2)  # BOE -> OBE
        h_object = self.transformer_object(h_object)
        h_object = h_object.permute(1, 0, 2)  # OBE -> BOE

        h_object = h_object.mean(dim=1)

        h_action = self.fc_action(actions)

        h_object = h_object[:, None, :].repeat(1, A, 1)
        h_action = h_action[None, :, :].repeat(B, 1, 1)

        h = torch.cat([h_object, h_action], dim=2)

        h = h.reshape(B * A, -1)
        h = self.fc_output(h)
        assert h.shape[-1] == 1
        h = h.reshape(B, -1)

        return h
