"""Comprehensive evaluation metrics for flood mapping.

10 binary metrics + 3-class BlackBench-compatible metrics:

Binary (flood vs non-flood):
1. IoU (water)       - Intersection over Union for flood class
2. F1 Score           - Harmonic mean of precision and recall
3. Precision          - TP / (TP + FP)
4. Recall             - TP / (TP + FN)
5. mIoU (macro)       - Mean IoU across flood and non-flood classes
6. Kappa              - Cohen's kappa coefficient
7. Betti0 Error       - Connected component count difference
8. Betti1 Error       - Hole count difference
9. HVR               - Height violation rate (flood above HAND threshold)
10. PA-IoU           - Physics-Aware IoU (mIoU * (1 - HVR))

3-class (BlackBench-compatible):
- F1-NW: F1 for No Water class
- F1-PW: F1 for Permanent Water class
- F1-F:  F1 for Flood class
- mIoU:  Mean IoU across all 3 classes
- F1-W:  Binary F1 for Water (PW + Flood)
"""

import numpy as np
import torch
from scipy.ndimage import label


def count_betti1(binary_mask, n_max=1000):
    """Count number of holes (Betti-1) in a binary mask."""
    inverted = ~binary_mask.astype(bool)
    struct_4conn = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labeled, n = label(inverted, structure=struct_4conn)
    if n > n_max:
        return 0
    border_labels = set(np.unique(np.concatenate([
        labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]
    ])))
    internal_holes = sum(1 for l in range(1, n + 1) if l not in border_labels)
    return internal_holes


def compute_confusion(pred, target):
    """Compute TP, FP, FN, TN."""
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    return tp, fp, fn, tn


def compute_all_metrics(pred_prob, target, raw_hand=None, threshold=0.5, hand_thresh=5.0):
    """Compute all 10 evaluation metrics.

    Args:
        pred_prob: (B, 1, H, W) predicted probabilities.
        target: (B, 1, H, W) binary ground truth.
        raw_hand: (B, 1, H, W) raw HAND values in meters.
        threshold: Probability threshold for binarization.
        hand_thresh: HAND threshold for HVR (meters).

    Returns:
        dict with all metrics.
    """
    pred = (pred_prob > threshold).float()
    B = pred.shape[0]

    # Per-batch metrics
    water_ious = []
    f1_scores = []
    precisions = []
    recalls = []
    mious = []
    kappas = []
    b0_errs = []
    b1_errs = []
    hvrs = []
    paious = []

    for i in range(B):
        p = pred[i, 0]
        t = target[i, 0]

        tp, fp, fn, tn = compute_confusion(p, t)

        # 1. IoU (water)
        iou = (tp) / (tp + fp + fn + 1e-6)
        water_ious.append(iou)

        # 2. F1 Score
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        f1_scores.append(f1)

        # 3. Precision
        precisions.append(precision)

        # 4. Recall
        recalls.append(recall)

        # 5. mIoU (macro average)
        bg_iou = (tn) / (tn + fp + fn + 1e-6)
        miou = (iou + bg_iou) / 2.0
        mious.append(miou)

        # 6. Cohen's Kappa
        n = tp + fp + fn + tn
        p_obs = (tp + tn) / n
        p_exp = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
        kappa = (p_obs - p_exp) / (1.0 - p_exp + 1e-6)
        kappas.append(kappa)

        # 7. Betti-0 Error (component count)
        p_np = p.detach().cpu().numpy().astype(np.uint8)
        t_np = t.detach().cpu().numpy().astype(np.uint8)
        _, p_b0 = label(p_np)
        _, t_b0 = label(t_np)
        b0_errs.append(min(abs(p_b0 - t_b0), 50))

        # 8. Betti-1 Error (hole count)
        b1_errs.append(min(abs(count_betti1(p_np) - count_betti1(t_np)), 20))

        # 9. HVR (Height Violation Rate)
        if raw_hand is not None:
            rh = raw_hand[i, 0]
            violations = ((p == 1.0) & (rh > hand_thresh)).sum().item()
            hvr = violations / (p.sum().item() + 1e-6)
        else:
            hvr = 0.0
        hvrs.append(hvr)

        # 10. PA-IoU (Physics-Aware IoU)
        paiou = miou * (1.0 - hvr)
        paious.append(paiou)

    # Average across batch
    metrics = {
        'IoU': float(np.mean(water_ious)),
        'F1': float(np.mean(f1_scores)),
        'Precision': float(np.mean(precisions)),
        'Recall': float(np.mean(recalls)),
        'mIoU': float(np.mean(mious)),
        'Kappa': float(np.mean(kappas)),
        'Betti0_Err': float(np.mean(b0_errs)),
        'Betti1_Err': float(np.mean(b1_errs)),
        'HVR': float(np.mean(hvrs)),
        'PA_IoU': float(np.mean(paious)),
        'IoU_std': float(np.std(water_ious)),
        'F1_std': float(np.std(f1_scores)),
    }

    return metrics


# Notebook compatibility alias.
compute_metrics = compute_all_metrics


def compute_3class_metrics(logits_3class, target_3class):
    """Compute BlackBench-compatible 3-class semantic segmentation metrics.

    Uses the model's 3-class output head (softmax over NW, PW, Flood)
    exactly matching BlackBench's evaluation methodology.

    Args:
        logits_3class: (B, 3, H, W) raw logits for 3 classes [NW, PW, Flood].
        target_3class: (B, H, W) long tensor with values 0=NW, 1=PW, 2=Flood.

    Returns:
        dict with F1_NW, F1_PW, F1_F, mIoU_3class, F1_W (binary water),
        and per-class IoU values matching BlackBench Table 2 format.
    """
    B = logits_3class.shape[0]
    pred = logits_3class.argmax(dim=1).long()  # (B, H, W)
    target = target_3class
    if target.dim() == 4 and target.shape[1] == 1:
        target = target[:, 0]  # (B, 1, H, W) -> (B, H, W)

    ious = {0: [], 1: [], 2: []}
    f1s = {0: [], 1: [], 2: []}
    f1_w_list = []

    for i in range(B):
        for cls in [0, 1, 2]:
            p = (pred[i] == cls).float()
            t = (target[i] == cls).float()
            tp = (p * t).sum().item()
            fp = (p * (1 - t)).sum().item()
            fn = ((1 - p) * t).sum().item()
            iou = (tp + 1e-8) / (tp + fp + fn + 1e-8)
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            ious[cls].append(iou)
            f1s[cls].append(f1)

        # Binary water (PW + Flood vs NW) — matches F1-W in BlackBench
        p_water = (pred[i] >= 1).float()
        t_water = (target[i] >= 1).float()
        tp_w = (p_water * t_water).sum().item()
        fp_w = (p_water * (1 - t_water)).sum().item()
        fn_w = ((1 - p_water) * t_water).sum().item()
        prec_w = tp_w / (tp_w + fp_w + 1e-8)
        rec_w = tp_w / (tp_w + fn_w + 1e-8)
        f1_w = 2 * prec_w * rec_w / (prec_w + rec_w + 1e-8)
        f1_w_list.append(f1_w)

    # Average across batch
    metrics = {
        # Per-class F1 (matching BlackBench's F1-NW, F1-PW, F1-F)
        'F1_NW': float(np.mean(f1s[0])) * 100,
        'F1_PW': float(np.mean(f1s[1])) * 100,
        'F1_F': float(np.mean(f1s[2])) * 100,
        # Per-class IoU
        'IoU_NW': float(np.mean(ious[0])) * 100,
        'IoU_PW': float(np.mean(ious[1])) * 100,
        'IoU_F': float(np.mean(ious[2])) * 100,
        # Mean IoU across all 3 classes (exactly matching BlackBench)
        'mIoU_3class': float(np.mean([np.mean(ious[c]) for c in [0, 1, 2]])) * 100,
        # Binary water F1 (matching BlackBench's F1-W)
        'F1_W': float(np.mean(f1_w_list)) * 100,
    }
    return metrics


# Alias for notebook compatibility
compute_multiclass_metrics = compute_3class_metrics
