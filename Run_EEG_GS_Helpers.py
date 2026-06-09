import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianEEGField(nn.Module):
    """
    GS-style EEG field.

    Input:
        points: [N, 4] = normalized [x, y, z, t]

    Output:
        voltage: [N, 1]
    """

    def __init__(self, centers, hidden_dim=64, learn_centers=False):
        super().__init__()

        # centers: [K, 3] normalized electrode positions
        if learn_centers:
            self.centers = nn.Parameter(centers.clone().float())
        else:
            self.register_buffer("centers", centers.clone().float())

        self.K = centers.shape[0]

        # One spatial scale per Gaussian and per axis: [K, 3]
        self.log_scales = nn.Parameter(torch.full((self.K, 3), -1.5))

        # Importance of each Gaussian: [K]
        self.alpha_logits = nn.Parameter(torch.zeros(self.K))

        # Temporal network:
        # gets t and predicts one amplitude per Gaussian
        self.time_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.K)
        )

    def forward(self, points):
        xyz = points[:, :3]       # [N, 3]
        t = points[:, 3:4]        # [N, 1]

        scales = F.softplus(self.log_scales) + 1e-4      # [K, 3]
        alpha = F.softplus(self.alpha_logits) + 1e-8     # [K]

        # diff: [N, K, 3]
        diff = xyz[:, None, :] - self.centers[None, :, :]

        # Gaussian distance: [N, K]
        dist2 = torch.sum((diff / scales[None, :, :]) ** 2, dim=-1)

        # Spatial Gaussian weights: [N, K]
        spatial_weights = torch.exp(-0.5 * dist2) * alpha[None, :]

        # Normalize weights so prediction stays stable
        spatial_weights = spatial_weights / (
            spatial_weights.sum(dim=1, keepdim=True) + 1e-8
        )

        # Time-dependent Gaussian amplitudes: [N, K]
        temporal_amplitudes = self.time_net(t)

        # Final EEG voltage: [N, 1]
        voltage = torch.sum(spatial_weights * temporal_amplitudes, dim=1, keepdim=True)

        return voltage
    