"""Physics-constrained loss functions for D8FN.

Novel loss components:
1. Flow directional derivative: ∇H_w · flow_dir (proper vector-based)
2. Flow accumulation continuity: height above DEM correlates with accumulation
3. Height consistency: H_w >= DEM for flooded pixels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(logits, targets, gamma=2.0, alpha=0.25, reduction='mean'):
    """Focal Loss for binary segmentation.

    FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)
    where p_t = σ(logits) for positives, 1-σ(logits) for negatives.

    Args:
        logits: (B, 1, H, W) raw logits
        targets: (B, 1, H, W) binary {0, 1}
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Balancing factor for positives
        reduction: 'mean' or 'sum'

    Returns:
        Scalar loss value
    """
    probs = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    # Focal weighting: (1-p_t)^gamma
    p_t = targets * probs + (1 - targets) * (1 - probs)
    focal_weight = (1 - p_t) ** gamma

    # Alpha balancing
    alpha_weight = targets * alpha + (1 - targets) * (1 - alpha)

    loss = alpha_weight * focal_weight * ce_loss
    return loss.mean() if reduction == 'mean' else loss.sum()


class D8FNLoss(nn.Module):
    """Physics-constrained loss for D8FN height field prediction.

    Combines Focal Loss (replaces BCE), Dice Loss, and novel physics constraints
    derived from the differentiable D8 flow graph.
    """

    # D8 unit vectors
    DX = torch.tensor([0., 1., 0.7071, 0., -0.7071, -1., -0.7071, 0., 0.7071])
    DY = torch.tensor([0., 0., -0.7071, -1., -0.7071, 0., 0.7071, 1., 0.7071])

    def __init__(self, physics_weight=1.0, focal_gamma=2.0, focal_alpha=0.75):
        super().__init__()
        self.physics_weight = physics_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # Register buffers
        self.register_buffer('_dx', self.DX)
        self.register_buffer('_dy', self.DY)

    def forward(self, logits, targets, H_w, dem_raw, hand, flow_dir, flow_acc,
                logits_3class=None, label_3class=None,
                return_components=False):
        """
        Args:
            logits: (B, 1, H, W) flood logits
            targets: (B, 1, H, W) binary flood mask
            H_w: (B, 1, H, W) predicted water height
            dem_raw: (B, 1, H, W) digital elevation model
            hand: (B, 1, H, W) height above nearest drainage
            flow_dir: (B, 1, H, W) D8 flow direction (0-1 normalized)
            flow_acc: (B, 1, H, W) flow accumulation (log-normalized)
            return_components: if True, return dict of loss components

        Returns:
            total_loss or (total_loss, components_dict)
        """
        probs = torch.sigmoid(logits)
        B, _, H, W = targets.shape

        # ---- 1. Focal Loss (replaces standard BCE) ----
        focal = focal_loss(logits, targets, gamma=self.focal_gamma, alpha=self.focal_alpha)

        # ---- 2. Dice Loss ----
        intersect = (probs * targets).sum(dim=(2, 3))
        card = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = 1.0 - ((2.0 * intersect + 1e-6) / (card + 1e-6)).mean()

        # ---- 3. Height Consistency (light regularizer) ----
        # Normalize raw DEM to match H_w scale [0,1] (same normalization as data.py)
        dem_norm = (torch.clamp(dem_raw, -50.0, 3000.0) + 50.0) / 3050.0
        margin = 0.01
        height_violations = F.relu(dem_norm - H_w + margin)
        height_loss = (height_violations * targets).mean() * 0.3

        # ---- 4. Flow Directional Derivative (light regularizer) ----
        dir_idx = (flow_dir * 8.0).round().long().clamp_(0, 8)

        fd_x = self._dx.to(flow_dir.device)[dir_idx]
        fd_y = self._dy.to(flow_dir.device)[dir_idx]

        dH_dx = torch.zeros_like(H_w)
        dH_dy = torch.zeros_like(H_w)
        dH_dx[:, :, :, 1:-1] = (H_w[:, :, :, 2:] - H_w[:, :, :, :-2]) * 0.5
        dH_dy[:, :, 1:-1, :] = (H_w[:, :, 2:, :] - H_w[:, :, :-2, :]) * 0.5

        directional_deriv = dH_dx * fd_x + dH_dy * fd_y
        flow_consistency = F.relu(directional_deriv).mean() * 0.2

        # ---- 5. Flow Continuity Prior (light regularizer) ----
        height_above_dem = F.relu(H_w - dem_norm)
        with torch.no_grad():
            h_scale = height_above_dem.flatten(1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
            f_scale = flow_acc.flatten(1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
        h_norm = height_above_dem / (h_scale + 1e-6)
        f_norm = flow_acc / (f_scale + 1e-6)
        continuity_prior = F.softplus(-(h_norm * f_norm).mean()) * 0.1

        # ---- 6. Height Smoothness (light) ----
        dy = torch.abs(H_w[:, :, 1:, :] - H_w[:, :, :-1, :]).mean()
        dx = torch.abs(H_w[:, :, :, 1:] - H_w[:, :, :, :-1]).mean()
        smoothness = (dy + dx) * 0.01

        # ---- 7. Boundary-Focused Focal (moderate) ----
        boundary = torch.abs(F.avg_pool2d(targets, 3, 1, 1) - targets)
        full_focal = ((1.0 + 5.0 * boundary) * 
            F.binary_cross_entropy_with_logits(logits, targets, reduction='none')).mean() * 0.5

        # ---- 8. HAND Physics Penalty (light regularizer) ----
        physics_penalty = (probs * hand).mean() * 0.2

        # ---- 9. 3-Class Cross-Entropy ----
        ce_3d = 0.0
        if logits_3class is not None and label_3class is not None:
            ce_3d = F.cross_entropy(
                logits_3class, label_3class[:, 0].long(), ignore_index=-1
            ) * 0.5

        # Combine: seg ~3.0, physics ~0.8 (seg leads 4:1, physics is light regularizer)
        total = (focal + dice + height_loss + flow_consistency + continuity_prior
                 + smoothness + full_focal + physics_penalty + ce_3d)

        if return_components:
            components = {
                'focal': focal.item(),
                'dice': dice.item(),
                'height_consistency': height_loss.item(),
                'flow_directional': flow_consistency.item(),
                'continuity': continuity_prior.item(),
                'smoothness': smoothness.item(),
                'full_focal': full_focal.item(),
                'physics_penalty': physics_penalty.item(),
                'ce_3d': ce_3d.item() if isinstance(ce_3d, torch.Tensor) and ce_3d.numel() == 1 else 0.0,
                'total': total.item(),
            }
            return total, components

        return total


class BCEDiceLoss(nn.Module):
    """Standard BCE + Dice loss for baseline models (no physics)."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.register_buffer('_pw', torch.tensor([5.0]))

    def forward(self, logits, targets, logits_3class=None, label_3class=None, return_components=False):
        probs = torch.sigmoid(logits)
        self.bce.pos_weight = self._pw.to(logits.device)
        bce = self.bce(logits, targets).mean()
        intersect = (probs * targets).sum(dim=(2, 3))
        card = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = 1.0 - ((2.0 * intersect + 1e-6) / (card + 1e-6)).mean()
        boundary = torch.abs(F.avg_pool2d(targets, 3, 1, 1) - targets)
        focal = ((1.0 + 5.0 * boundary) * self.bce(logits, targets)).mean()
        total = bce + dice + 0.5 * focal
        
        ce_3d = 0.0
        if logits_3class is not None and label_3class is not None:
            ce_3d = F.cross_entropy(
                logits_3class, label_3class[:, 0].long(), ignore_index=-1
            ) * 0.5
            total += ce_3d

        if return_components:
            return total, {'bce': bce.item(), 'dice': dice.item(), 'ce_3d': ce_3d.item() if isinstance(ce_3d, torch.Tensor) else 0.0, 'total': total.item()}
        return total
