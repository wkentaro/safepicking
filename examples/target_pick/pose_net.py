import torch


class PoseNet(torch.nn.Module):
    def __init__(self, closedloop, n_action, **kwargs):
        super().__init__()

        self.closedloop = closedloop

        # ee_pose/grasp_pose: 7, n_past_action * 7
        in_channels = 7 + kwargs["n_past_action"] * 7
        self.fc_context = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 10),
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

        if self.closedloop:
            h_context = [
                kwargs["ee_pose"][batch_indices],
                kwargs["past_grasped_object_poses"][batch_indices],
            ]
        else:
            h_context = [
                kwargs["grasp_pose"][batch_indices],
                kwargs["past_grasped_object_poses"][batch_indices],
            ]
        h_context = torch.cat(h_context, dim=1)
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
