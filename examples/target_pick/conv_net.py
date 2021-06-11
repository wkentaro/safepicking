import torch


class ConvNet(torch.nn.Module):
    def __init__(self, episode_length):
        super().__init__()

        # heightmap: 1
        self.encoder_heightmap = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )
        # maskmap: 1
        self.encoder_maskmap = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(4 + 4, 4, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )

        # h: 32
        # actions: 6
        # ee_poses: (episode_length - 1) * 7
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(32 + 6 + (episode_length - 1) * 7, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
        )

    def forward(self, heightmap, maskmap, grasped_uv, ee_poses, actions):
        B = heightmap.shape[0]
        A = actions.shape[0]

        h_heightmap = self.encoder_heightmap(heightmap[:, None, :, :])
        h_maskmap = self.encoder_maskmap(maskmap[:, None, :, :].float())
        h = torch.cat([h_heightmap, h_maskmap], dim=1)

        h = self.encoder(h)

        grasped_uv = torch.floor(grasped_uv.float() / 8).long()
        u = grasped_uv[:, 0]
        v = grasped_uv[:, 1]

        h = h[torch.arange(B), :, v, u]

        h = h[:, None, :].repeat_interleave(A, dim=1)
        h_actions = actions[None, :, :].repeat_interleave(B, dim=0)
        h_ee_poses = ee_poses[:, None, :, :].repeat_interleave(A, dim=1)
        h_ee_poses = h_ee_poses.reshape(B, A, -1)

        h = torch.cat([h, h_actions, h_ee_poses], dim=2)

        h = h.reshape(B * A, -1)

        h = self.mlp(h)

        h = h.reshape(B, A, 2)

        return h
