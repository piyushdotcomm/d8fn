"""Model architectures for D8FN and baselines."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm

from .routing import D8FlowRouting, D8FlowRoutingBlock


class FourierFeatureEncoder(nn.Module):
    """Positional encoding using Fourier features (NeRF-style)."""

    def __init__(self, in_features=2, num_frequencies=10):
        super().__init__()
        freqs = 2.0 ** torch.linspace(0.0, num_frequencies - 1, num_frequencies)
        self.register_buffer('freq_bands', freqs)

    def forward(self, x):
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq * math.pi))
            out.append(torch.cos(x * freq * math.pi))
        return torch.cat(out, dim=-1)


class HeightFieldHead(nn.Module):
    """Decodes a continuous height field from features.

    Predicts water surface height H_w and flood probability.
    """

    def __init__(self, feat_dim=128, hidden_dim=256):
        super().__init__()
        self.fourier = FourierFeatureEncoder(in_features=2, num_frequencies=10)
        fourier_dim = 42  # 2 input + 10 freqs * 2 (sin, cos)

        self.height_mlp = nn.Sequential(
            nn.Linear(feat_dim + fourier_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.tau = nn.Parameter(torch.tensor(0.5))

    def forward(self, feat_map, dem_raw):
        B, C, H, W = feat_map.shape
        device = feat_map.device

        # Grid coordinates in [-1, 1]
        gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        coords = torch.stack([gx, gy], dim=-1)
        coord_enc = self.fourier(coords).permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)

        # Concatenate features with coordinates and DEM
        combined = torch.cat([feat_map, coord_enc, dem_raw], dim=1)
        flat = combined.permute(0, 2, 3, 1).reshape(B * H * W, -1)

        # Predict water height
        H_w = self.height_mlp(flat).view(B, 1, H, W)

        # Flood probability = P(H_w > DEM) via sigmoid with learned temperature
        tau_pos = F.softplus(self.tau) + 0.01
        flood_logit = (H_w - dem_raw) / tau_pos

        return flood_logit, H_w


# ============================================================
# Progressive Decoder with Skip Connections
# ============================================================

class DecoderWithSkips(nn.Module):
    """Decoder starting from 28×28 (after fine routing at 28×28).

    Routes: 28×28 → 56×56 (skip from encoder stage 0) → 112×112 → 224×224.
    """
    def __init__(self, feat_dims, final_dim=128):
        super().__init__()
        f56, f28, _, _ = feat_dims  # [96, 192] — stage 0 (56×56) and stage 1 (28×28)

        # Fuse input (routed 28×28 features + encoder stage 1 features)
        self.fuse_input = nn.Sequential(
            nn.Conv2d(256 + f28, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.SiLU(True),
        )

        # 28→56 upsampling → fuse with encoder stage 0 (56x56, 96ch)
        self.up_28_56 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128), nn.SiLU(True),
        )
        self.fuse_56 = nn.Sequential(
            nn.Conv2d(128 + f56, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(True),
        )

        # 56→224 final upsampling (4×)
        self.up_to_full = nn.Sequential(
            nn.ConvTranspose2d(64, final_dim, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(final_dim), nn.SiLU(True),
        )

    def forward(self, feat_28, encoder_feats):
        """Decode from 28×28 features with skip connections.

        Args:
            feat_28: (B, 256, 28, 28) — fused routing output at 28×28
            encoder_feats: [(B, 96, 56, 56), (B, 192, 28, 28)]
        """
        f28e = encoder_feats[1]  # (B, 192, 28, 28)
        f56e = encoder_feats[0]  # (B, 96, 56, 56)

        x = self.fuse_input(torch.cat([feat_28, f28e], dim=1))  # (B, 128, 28, 28)
        x = self.up_28_56(x)                                      # (B, 128, 56, 56)
        x = self.fuse_56(torch.cat([x, f56e], dim=1))             # (B, 64, 56, 56)
        x = self.up_to_full(x)                                    # (B, 128, 224, 224)
        return x


# ============================================================
# D8FN: The Novel Architecture (Improved)
# ============================================================

class D8FN(nn.Module):
    """Differentiable D8 Flow Graph Network — Improved version.

    Key novelties:
    1. D8 flow routing at multiple scales (7x7 coarse + 14x14 fine)
    2. Progressive decoder with skip connections from all encoder stages
    3. Physics-constrained height field prediction

    Architecture:
        SAR+DEM → ConvNeXt Encoder → Multi-scale D8 Routing
        → Progressive Decoder (with skips) → HeightFieldHead + 3-class Head
    """

    def __init__(self, in_ch=9, backbone='convnext_small.fb_in22k_ft_in1k_384',
                 routing_rounds=50, routing_dim=64, height_dim=256):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_ch, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3), nn.SiLU(True),
        )

        # Encoder — all 4 feature levels
        self.encoder = timm.create_model(
            backbone, pretrained=True, features_only=True,
            out_indices=[0, 1, 2, 3]
        )
        feat_dims = [f['num_chs'] for f in self.encoder.feature_info]  # [96, 192, 384, 768]

        # Block 1: Coarse routing at 7×7 (50 rounds)
        self.flow_routing_coarse = D8FlowRoutingBlock(
            feat_dims[3], hidden_dim=routing_dim, num_rounds=routing_rounds
        )

        # Block 2: Fine routing at 28×28 (25 rounds) — captures meaningful flow paths
        self.flow_routing_fine = D8FlowRoutingBlock(
            feat_dims[1], hidden_dim=routing_dim, num_rounds=routing_rounds // 2
        )

        # Projection: coarse 7×7 (768ch) → fine 28×28 level (192ch)
        self._proj_7_to_28 = nn.Sequential(
            nn.Conv2d(feat_dims[3], feat_dims[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dims[1]), nn.SiLU(True),
        )

        # Fuse coarse (192ch at 28×28) + fine (192ch at 28×28) into 256ch
        self.fuse_multiscale = nn.Sequential(
            nn.Conv2d(feat_dims[1] * 2, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.SiLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.SiLU(True),
        )

        # Decoder: 28→56→224 with skip connections
        self.decoder = DecoderWithSkips(feat_dims, final_dim=128)

        # Heads
        self.height_head = HeightFieldHead(feat_dim=128, hidden_dim=height_dim)
        self.head_3class = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1), nn.SiLU(True),
            nn.Conv2d(64, 3, kernel_size=1),
        )

    def forward(self, sar, dem, hand, slope, flow_dir, flow_acc):
        x = torch.cat([sar, dem, hand, slope, flow_dir, flow_acc], dim=1)
        x = self.input_proj(x)

        enc = self.encoder(x)  # [56×56, 28×28, 14×14, 7×7]
        f7 = enc[-1]           # (B, 768, 7, 7)

        # Coarse routing at 7×7
        f7_routed = self.flow_routing_coarse(f7, flow_dir, dem, slope)

        # Project and upsample 7×7 → 28×28
        F_in = F.interpolate
        f7_proj = self._proj_7_to_28(F_in(f7_routed, size=enc[1].shape[2:], mode='bilinear', align_corners=False))

        # Fine routing at 28×28
        f28_routed = self.flow_routing_fine(enc[1], flow_dir, dem, slope)

        # Fuse multi-scale routing outputs
        f_fused = self.fuse_multiscale(torch.cat([f28_routed, f7_proj], dim=1))  # (B, 256, 28, 28)

        # Decode to 224×224 with skip connections
        feat_map = self.decoder(f_fused, enc[:2])  # (B, 128, 224, 224)

        dem_f = F_in(dem, size=(224, 224), mode='bilinear', align_corners=False)
        flood_logit, H_w = self.height_head(feat_map, dem_f)
        logits_3class = self.head_3class(feat_map)

        return flood_logit, H_w, logits_3class


class D8FN_Light(D8FN):
    """Lightweight D8FN for ablation studies."""

    def __init__(self, in_ch=9):
        super().__init__(
            in_ch=in_ch,
            backbone='convnext_tiny.fb_in22k_ft_in1k_384',
            routing_rounds=25,
            routing_dim=32,
            height_dim=128,
        )


# ============================================================
# D8FN without flow routing (ablation: remove the novel component)
# ============================================================

class D8FN_NoRouting(D8FN):
    """D8FN without D8 flow routing (ablation control)."""

    def __init__(self, in_ch=9):
        super().__init__(in_ch=in_ch)
        class _ConvAblationBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channels), nn.SiLU(True),
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                )
                self.fuse = nn.Sequential(
                    nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels), nn.SiLU(True),
                )

            def forward(self, feat, *args, **kwargs):
                return self.fuse(torch.cat([self.conv(feat), feat], dim=1)) + feat

        feat_dims = [f['num_chs'] for f in self.encoder.feature_info]
        self.flow_routing_coarse = _ConvAblationBlock(feat_dims[3])
        self.flow_routing_fine = _ConvAblationBlock(feat_dims[1])


class D8FN_NoPhysicsLoss(D8FN):
    """D8FN without physics-constrained losses (ablation control).

    Uses only standard BCE+Dice loss.
    """
    pass  # Handled at loss level


# ============================================================
# BASELINES: Standard architectures for comparison
# ============================================================

class UNetBaseline(nn.Module):
    """U-Net baseline (matches BlackBench best model, 3-class output)."""

    def __init__(self, in_ch=9):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.model = smp.Unet(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            in_channels=in_ch,
            classes=3,  # 3-class output: NW, PW, Flood
            activation=None,
        )

    def forward(self, sar, dem, hand, slope, flow_dir, flow_acc):
        x = torch.cat([sar, dem, hand, slope, flow_dir, flow_acc], dim=1)
        logits_3class = self.model(x)
        flood_logit = logits_3class[:, 2:3]  # Flood class (index 2) for BCE loss
        return flood_logit, None, logits_3class


class UNetBaseline_Small(nn.Module):
    """Smaller U-Net baseline with 3-class output."""

    def __init__(self, in_ch=9):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.model = smp.Unet(
            encoder_name='resnet18',
            encoder_weights='imagenet',
            in_channels=in_ch,
            classes=3,
            activation=None,
        )

    def forward(self, sar, dem, hand, slope, flow_dir, flow_acc):
        x = torch.cat([sar, dem, hand, slope, flow_dir, flow_acc], dim=1)
        logits_3class = self.model(x)
        flood_logit = logits_3class[:, 2:3]
        return flood_logit, None, logits_3class


class DeepLabBaseline(nn.Module):
    """DeepLabV3+ baseline with 3-class output."""

    def __init__(self, in_ch=9):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.model = smp.DeepLabV3Plus(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            in_channels=in_ch,
            classes=3,
            activation=None,
        )

    def forward(self, sar, dem, hand, slope, flow_dir, flow_acc):
        x = torch.cat([sar, dem, hand, slope, flow_dir, flow_acc], dim=1)
        logits_3class = self.model(x)
        flood_logit = logits_3class[:, 2:3]
        return flood_logit, None, logits_3class


class UNetConvNeXt(nn.Module):
    """Fair-comparison baseline: standard U-Net decoder on ConvNeXt-Small backbone.

    Uses the SAME backbone as D8FN (convnext_small.fb_in22k_ft_in1k_384) so any
    performance delta vs D8FN is attributable to the D8 routing architecture alone,
    not to backbone capacity differences.

    Decoder: bilinear upsample + skip connections (no D8 routing, no physics loss).
    Output: (flood_logit [B,1,H,W], None, logits_3class [B,3,H,W])
    """

    def __init__(self, in_ch: int = 9,
                 backbone: str = 'convnext_small.fb_in22k_ft_in1k_384'):
        super().__init__()

        # ── Encoder: ConvNeXt-Small, identical to D8FN ───────────────────
        self.encoder = timm.create_model(
            backbone, pretrained=True, features_only=True,
            out_indices=[0, 1, 2, 3], in_chans=in_ch
        )
        # ConvNeXt-Small feature dims: [96, 192, 384, 768]
        C = [f['num_chs'] for f in self.encoder.feature_info]

        # ── Decoder (U-Net style, bilinear + skip connections) ───────────
        # enc[3] 7×7 768 → upsample 2× → cat enc[2] 14×14 384 → 256ch
        self.dec3 = self._block(C[3] + C[2], 256)
        # dec3 14×14 256 → upsample 2× → cat enc[1] 28×28 192 → 128ch
        self.dec2 = self._block(256 + C[1], 128)
        # dec2 28×28 128 → upsample 2× → cat enc[0] 56×56 96 → 64ch
        self.dec1 = self._block(128 + C[0], 64)
        # dec1 56×56 64 → upsample 4× → 224×224 → 64ch
        self.dec0 = self._block(64, 64)

        # ── Output heads (same as D8FN) ───────────────────────────────────
        self.flood_head = nn.Sequential(
            nn.Conv2d(64, 32, 1), nn.SiLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        self.head_3class = nn.Sequential(
            nn.Conv2d(64, 32, 1), nn.SiLU(inplace=True),
            nn.Conv2d(32, 3, 1),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True),
        )

    def forward(self, sar, dem, hand, slope, flow_dir, flow_acc):
        x = torch.cat([sar, dem, hand, slope, flow_dir, flow_acc], dim=1)
        enc = self.encoder(x)  # [56×56, 28×28, 14×14, 7×7]

        # Decoder with skip connections
        f = F.interpolate(enc[3], size=enc[2].shape[2:],
                          mode='bilinear', align_corners=False)
        f = self.dec3(torch.cat([f, enc[2]], dim=1))   # 14×14, 256ch

        f = F.interpolate(f, size=enc[1].shape[2:],
                          mode='bilinear', align_corners=False)
        f = self.dec2(torch.cat([f, enc[1]], dim=1))   # 28×28, 128ch

        f = F.interpolate(f, size=enc[0].shape[2:],
                          mode='bilinear', align_corners=False)
        f = self.dec1(torch.cat([f, enc[0]], dim=1))   # 56×56, 64ch

        f = F.interpolate(f, size=(224, 224),
                          mode='bilinear', align_corners=False)
        f = self.dec0(f)                                # 224×224, 64ch

        return self.flood_head(f), None, self.head_3class(f)


# Model registry for experiment configurations
MODEL_REGISTRY = {
    'D8FN': D8FN,
    'D8FN_Light': D8FN_Light,
    'D8FN_NoRouting': D8FN_NoRouting,
    'UNet': UNetBaseline,
    'UNet_Small': UNetBaseline_Small,
    'DeepLab': DeepLabBaseline,
    'UNetConvNeXt': UNetConvNeXt,
}
