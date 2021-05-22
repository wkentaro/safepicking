import torch


class PoseNet(torch.nn.Module):
    def __init__(self, n_action):
        super().__init__()

        # object_labels: 7, object_poses: 7, grasp_flags: 1
        self.fc_encoder = torch.nn.Sequential(
            torch.nn.Linear(7 + 7 + 1, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=512,
            nhead=4,
            dim_feedforward=1024,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=4, norm=None
        )

        self.fc_output = torch.nn.Sequential(
            torch.nn.Linear(512, n_action),
        )

    def forward(
        self,
        object_labels,
        object_poses,
        grasp_flags,
        **kwargs,
    ):
        B, O = grasp_flags.shape

        is_valid = (object_labels > 0).any(dim=2)

        h = torch.cat(
            [object_labels, object_poses, grasp_flags[:, :, None]], dim=2
        )

        batch_indices = torch.where(is_valid.sum(dim=1))[0]

        h = h[batch_indices]
        h = h.reshape(batch_indices.shape[0] * O, h.shape[2])
        h = self.fc_encoder(h)
        h = h.reshape(batch_indices.shape[0], O, h.shape[1])  # BOE

        h = h.permute(1, 0, 2)  # BOE -> OBE
        h = self.transformer_encoder(h, src_key_padding_mask=~is_valid)
        h = h.permute(1, 0, 2)  # OBE -> BOE

        # mean
        is_valid = is_valid[:, :, None].float()
        assert not torch.isnan(is_valid).any().item()

        h = (is_valid * h).sum(dim=1) / is_valid.sum(dim=1)
        assert not torch.isnan(h).any().item()

        output = torch.zeros((B, h.shape[1]), dtype=h.dtype, device=h.device)
        output[batch_indices] = h
        assert not torch.isnan(h).any().item()

        output = self.fc_output(output)

        return output
