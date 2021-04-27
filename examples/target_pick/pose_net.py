import torch


class PoseNet(torch.nn.Module):
    def __init__(self, n_action):
        super().__init__()

        # ee_pose: 7
        self.fc_context = torch.nn.Sequential(
            torch.nn.Linear(7, 10),
            torch.nn.ReLU(),
        )

        # object_labels: 7, object_poses: 7, grasp_flags: 1
        self.fc_encoder = torch.nn.Sequential(
            torch.nn.Linear(7 + 7 + 1, 10),
            torch.nn.ReLU(),
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=20,
            nhead=4,
            dim_feedforward=40,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=2, norm=None
        )

        self.fc_output = torch.nn.Sequential(
            torch.nn.Linear(20, n_action),
        )

    def forward(
        self,
        grasp_position,
        past_actions,
        object_labels,
        object_poses,
        grasp_flags,
    ):
        B, O = grasp_flags.shape

        is_valid = (object_labels > 0).any(dim=2)

        h = torch.cat(
            [object_labels, object_poses, grasp_flags[:, :, None]], dim=2
        )

        batch_indices = torch.where(is_valid.sum(dim=1))[0]
        grasp_position = grasp_position[batch_indices]
        past_actions = past_actions[batch_indices]
        h = h[batch_indices]

        h = h.reshape(batch_indices.shape[0] * O, h.shape[2])
        h = self.fc_encoder(h)
        h = h.reshape(batch_indices.shape[0], O, h.shape[1])  # BOE

        h_context = torch.cat([grasp_position, past_actions], dim=1)
        h_context = self.fc_context(h_context)  # BE
        h_context = h_context[:, None, :].repeat_interleave(O, dim=1)

        h = torch.cat([h, h_context], dim=2)

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
