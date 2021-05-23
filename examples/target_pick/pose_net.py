import torch


class PoseNet(torch.nn.Module):
    def __init__(self, n_action):
        super().__init__()

        # object_labels: 7, object_poses: 7, grasp_flags: 1
        self.fc_encoder = torch.nn.Sequential(
            torch.nn.Linear(7 + 7 + 1, 32),
            torch.nn.ReLU(),
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=4, norm=None
        )

        self.fc_output = torch.nn.Sequential(
            torch.nn.Linear(32, n_action),
        )

    def forward(
        self,
        object_labels,
        object_poses,
        grasp_flags,
        **kwargs,
    ):
        B, O = grasp_flags.shape

        h = torch.cat(
            [object_labels, object_poses, grasp_flags[:, :, None]], dim=2
        )

        h = h.reshape(B * O, h.shape[2])
        h = self.fc_encoder(h)
        h = h.reshape(B, O, h.shape[1])  # BOE

        h = h.permute(1, 0, 2)  # BOE -> OBE
        h = self.transformer_encoder(h)
        h = h.permute(1, 0, 2)  # OBE -> BOE

        output = self.fc_output(h.mean(dim=1))

        return output
