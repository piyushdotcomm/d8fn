
"""D8FlowRouting: A differentiable D8 flow routing layer for neural networks.

This is the core novel contribution of the paper. The layer:
1. Takes per-pixel features + DEM + D8 flow direction
2. Computes a learned gate for each feature channel per pixel
3. Propagates gated features downstream along D8 flow paths
4. Multiple rounds allow features to propagate through entire watersheds

Unlike standard convolutions (isotropic) or graph neural networks (O(N^2) attention),
D8FlowRouting is O(N) and follows the actual topology of water flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class D8FlowRouting(nn.Module):
    """Differentiable D8 flow routing layer."""

    DX = torch.tensor([0, 1, 1, 0, -1, -1, -1, 0, 1])
    DY = torch.tensor([0, 0, -1, -1, -1, 0, 1, 1, 1])

    def __init__(self, channels, hidden_dim=64, num_rounds=50, dropout=0.1):
        super().__init__()
        self.channels = channels
        self.num_rounds = num_rounds

        self.gate_net = nn.Sequential(
            nn.Conv2d(channels + 3, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.routing_temp = nn.Parameter(torch.tensor(0.5))
        self.channel_weights = nn.Parameter(torch.ones(channels))

        self.output_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def _compute_downstream_indices(self, flow_dir, device, B, H, W):
        dir_idx = (flow_dir * 8.0 + 0.5).floor_().long().clamp_(0, 8)

        y = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, 1, H, W).contiguous()
        x = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, 1, H, W).contiguous()

        dir_idx = dir_idx.contiguous()
        dx = self.DX.to(device)[dir_idx]
        dy = self.DY.to(device)[dir_idx]
        ny = y + dy
        nx = x + dx

        valid = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W) & (dir_idx > 0)

        downstream_idx = (ny * W + nx).long()
        downstream_idx[~valid] = torch.arange(H * W, device=device).view(1, 1, H, W).expand(B, 1, H, W)[~valid]

        return downstream_idx, valid

    def forward(self, features, flow_dir, dem, slope=None):
        B, C, H, W = features.shape
        N = H * W
        device = features.device

        if flow_dir.shape[2:] != features.shape[2:]:
            flow_dir = F.interpolate(flow_dir, size=features.shape[2:], mode='nearest')
            dem = F.interpolate(dem, size=features.shape[2:], mode='bilinear', align_corners=False)
            if slope is not None:
                slope = F.interpolate(slope, size=features.shape[2:], mode='bilinear', align_corners=False)
        if slope is None:
            slope = torch.zeros_like(dem)
        gate_input = [features, dem, flow_dir]
        if slope is not None:
            gate_input.append(slope)
        gate_input = torch.cat(gate_input, dim=1)

        gates = self.gate_net(gate_input)
        gates = gates * torch.sigmoid(self.routing_temp)
        gates = gates * self.channel_weights.view(1, -1, 1, 1)

        downstream_idx, valid = self._compute_downstream_indices(flow_dir, device, B, H, W)

        feat_flat = features.view(B, C, N)
        gates_flat = gates.view(B, C, N)
        valid_flat = valid.view(B, 1, N).to(features.dtype)

        accumulated = feat_flat.clone()

        for _ in range(self.num_rounds):
            flow = gates_flat * accumulated
            downstream = torch.zeros_like(accumulated)
            downstream.scatter_reduce_(2, downstream_idx.view(B, 1, N).expand(-1, C, -1), flow, reduce='sum', include_self=False)
            torch.cuda.synchronize()
            accumulated = accumulated + downstream

        routed = accumulated.view(B, C, H, W)
        output = self.output_proj(torch.cat([features, routed], dim=1))
        return output


class D8FlowRoutingBlock(nn.Module):
    """Residual block with D8 flow routing integrated."""

    def __init__(self, channels, hidden_dim=64, num_rounds=50):
        super().__init__()

        self.conv_path = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.flow_path = D8FlowRouting(
            channels, hidden_dim=hidden_dim, num_rounds=num_rounds
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x, flow_dir, dem, slope=None):
        conv_out = self.conv_path(x)
        flow_out = self.flow_path(x, flow_dir, dem, slope)
        fused = self.fusion(torch.cat([conv_out, flow_out], dim=1))
        return fused + x
