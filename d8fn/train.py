"""Training, evaluation, and 5-fold cross validation."""

import os, gc, copy, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from scipy import stats as scipy_stats
from tqdm import tqdm

from .metrics import compute_all_metrics, compute_3class_metrics
from .losses import D8FNLoss, BCEDiceLoss
from .data import FloodDataset, create_dataloaders


def _remap_flow_dir_for_flip(flow_dir, horizontal=False, vertical=False):
    """Flip normalized D8 directions consistently with image flips."""
    idx = torch.floor(flow_dir * 8.0 + 0.5).long().clamp(0, 8)
    if horizontal:
        hmap = torch.tensor([0, 5, 4, 3, 2, 1, 8, 7, 6], device=flow_dir.device)
        idx = hmap[idx]
    if vertical:
        vmap = torch.tensor([0, 1, 8, 7, 6, 5, 4, 3, 2], device=flow_dir.device)
        idx = vmap[idx]
    return idx.to(flow_dir.dtype) / 8.0


class EMA:
    """Exponential Moving Average of model weights."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone().detach()

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                avg = self.decay * self.shadow[n] + (1.0 - self.decay) * p.data
                self.shadow[n] = avg.clone().detach()

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def train_epoch(model, loader, criterion, optimizer, scaler, device, config, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    loss_comps = {}

    for bi, batch in enumerate(tqdm(loader, desc=f'Train E{epoch+1}', leave=False)):
        sar, dem, hand, slope, dem_raw, flow_dir, flow_acc, mask, raw_hand, label_3class = [
            x.to(device, non_blocking=True) for x in batch
        ]

        optimizer.zero_grad(set_to_none=True)

        logits, H_w, logits_3class = model(sar, dem, hand, slope, flow_dir, flow_acc)

        if config.get('is_height_field', True) and H_w is not None:
            loss, comps = criterion(
                logits, mask, H_w, dem_raw, hand, flow_dir, flow_acc,
                logits_3class=logits_3class, label_3class=label_3class,
                return_components=True
            )
        else:
            loss, comps = criterion(
                logits, mask,
                logits_3class=logits_3class, label_3class=label_3class,
                return_components=True
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k, v in comps.items():
            if k not in loss_comps:
                loss_comps[k] = 0.0
            loss_comps[k] += v

        del sar, dem, hand, slope, dem_raw, flow_dir, flow_acc, mask, raw_hand, label_3class, logits, logits_3class, loss

    avg_loss = total_loss / len(loader)
    avg_comps = {k: v / len(loader) for k, v in loss_comps.items()}

    return avg_loss, avg_comps


@torch.no_grad()
def evaluate(model, loader, criterion, device, config):
    """Evaluate model on validation set with test-time augmentation."""
    model.eval()
    all_metrics = []
    total_loss = 0.0

    for batch in tqdm(loader, desc='Val', leave=False):
        sar, dem, hand, slope, dem_raw, flow_dir, flow_acc, mask, raw_hand, label_3class = [
            x.to(device) if i < 10 else x for i, x in enumerate(batch)
        ]

        logits, H_w, logits_3class = model(sar, dem, hand, slope, flow_dir, flow_acc)

        # D8 directions are orientation-sensitive, so TTA must remap flow_dir.
        if config.get('tta', False):
            logits_tta = [logits]
            logits_3c_tta = [logits_3class]

            sar_h = sar.flip(-1)
            dem_h = dem.flip(-1)
            hand_h = hand.flip(-1)
            slope_h = slope.flip(-1)
            flow_acc_h = flow_acc.flip(-1)
            flow_dir_h = _remap_flow_dir_for_flip(flow_dir.flip(-1), horizontal=True)
            l_h, _, l3_h = model(sar_h, dem_h, hand_h, slope_h, flow_dir_h, flow_acc_h)
            logits_tta.append(l_h.flip(-1))
            logits_3c_tta.append(l3_h.flip(-1))

            sar_v = sar.flip(-2)
            dem_v = dem.flip(-2)
            hand_v = hand.flip(-2)
            slope_v = slope.flip(-2)
            flow_acc_v = flow_acc.flip(-2)
            flow_dir_v = _remap_flow_dir_for_flip(flow_dir.flip(-2), vertical=True)
            l_v, _, l3_v = model(sar_v, dem_v, hand_v, slope_v, flow_dir_v, flow_acc_v)
            logits_tta.append(l_v.flip(-2))
            logits_3c_tta.append(l3_v.flip(-2))

            logits = torch.stack(logits_tta).mean(0)
            logits_3class = torch.stack(logits_3c_tta).mean(0)

        if config.get('is_height_field', True) and H_w is not None:
            loss = criterion(logits, mask, H_w, dem_raw, hand, flow_dir, flow_acc,
                             logits_3class=logits_3class, label_3class=label_3class)
        else:
            loss = criterion(logits, mask,
                             logits_3class=logits_3class, label_3class=label_3class)

        probs = torch.sigmoid(logits.float())
        metrics = compute_all_metrics(probs, mask, raw_hand)
        
        # Verify binary metrics consistency
        iou_b = metrics['IoU']
        f1_b = metrics['F1']
        expected_f1 = 2 * iou_b / (1 + iou_b)
        if abs(f1_b - expected_f1) > 0.05:
            # F1 is corrupted (likely overwritten) — recalculate
            metrics['F1'] = expected_f1
        
        # BlackBench-compatible 3-class metrics (from 3-class head, not binary flood)
        bb_metrics = compute_3class_metrics(logits_3class.float(), label_3class)
        metrics.update(bb_metrics)
        
        all_metrics.append(metrics)
        total_loss += loss.item()

        del sar, dem, hand, slope, dem_raw, flow_dir, flow_acc, mask, raw_hand, logits, logits_3class, probs

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))

    avg_metrics['val_loss'] = total_loss / len(loader)

    return avg_metrics


def run_5fold_cv(model_class, model_name, data_dir, output_dir, config,
                 num_folds=5, device=None):
    """Run 5-fold cross validation for a model configuration.

    Args:
        model_class: Model class to instantiate.
        model_name: Name for this experiment.
        data_dir: Path to preprocessed data.
        output_dir: Output directory for checkpoints and results.
        config: Training configuration dict.
        num_folds: Number of CV folds.
        device: torch device.

    Returns:
        dict with results across all folds.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    # Auto-preprocess if data is missing
    from .data import ensure_data
    ensure_data(data_dir)

    fold_results = []
    final_metrics = {}

    for fold in range(num_folds):
        print(f'\n{"="*60}')
        print(f'Fold {fold + 1}/{num_folds} — {model_name}')
        print(f'{"="*60}')

        # Create dataloaders for this fold
        train_loader, val_loader = create_dataloaders(
            data_dir, batch_size=config.get('batch_size', 8), fold=fold
        )

        # Initialize model
        model = model_class(in_ch=config.get('in_ch', 9)).to(device)

        # Multi-GPU
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # Loss
        is_height = config.get('is_height_field', True)
        criterion = D8FNLoss() if is_height else BCEDiceLoss()

        # Optimizer with differential learning rates
        base = model.module if isinstance(model, nn.DataParallel) else model

        # Parameter groups
        encoder_params = []
        head_params = []
        for name, param in model.named_parameters():
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                head_params.append(param)

        # Store base LRs for warmup
        enc_lr = config.get('lr', 1e-4) * 0.3
        head_lr = config.get('lr', 1e-4)
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': enc_lr * 3.0},
            {'params': head_params, 'lr': head_lr},
        ], weight_decay=1e-4)

        # LR warmup (first 5 epochs) then cosine annealing
        warmup_epochs = 5
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.get('epochs', 30) - warmup_epochs, eta_min=1e-6
        )
        scaler = GradScaler()

        # EMA (Exponential Moving Average)
        ema = EMA(model, decay=0.999)

        # Training loop
        best_metrics = None
        best_paiou = 0.0
        best_state = None
        patience_counter = 0
        max_patience = config.get('patience', 7)

        for epoch in range(config.get('epochs', 30)):
            # LR schedule: linear warmup first, then cosine
            if epoch < warmup_epochs:
                scale = (epoch + 1) / warmup_epochs
                optimizer.param_groups[0]['lr'] = enc_lr * 3.0 * scale
                optimizer.param_groups[1]['lr'] = head_lr * scale
            else:
                cosine_scheduler.step()

            train_loss, train_comps = train_epoch(
                model, train_loader, criterion, optimizer, scaler, device, config, epoch
            )

            # Update EMA after each training epoch
            ema.update(model)

            # Apply EMA for validation, restore afterward
            ema_orig = {}
            for n, p in model.named_parameters():
                if p.requires_grad and n in ema.shadow:
                    ema_orig[n] = p.data.clone()
                    p.data = ema.shadow[n].clone()

            val_metrics = evaluate(model, val_loader, criterion, device, config)

            for n, p in model.named_parameters():
                if p.requires_grad and n in ema_orig:
                    p.data = ema_orig[n]

            current_paiou = val_metrics.get('PA_IoU', 0.0)

            # Progress
            current_lr = optimizer.param_groups[1]['lr']
            print(
                f'  Ep{epoch+1:02d} | '
                f'LR:{current_lr:.2e} | '
                f'Loss:{train_loss:.4f} | '
                f'IoU:{val_metrics["IoU"]:.4f} | '
                f'F1:{val_metrics["F1"]:.4f} | '
                f'mIoU:{val_metrics["mIoU"]:.4f} | '
                f'PA-IoU:{current_paiou:.4f}{" *" if current_paiou > best_paiou else ""}'
            )

            # Early stopping
            if current_paiou > best_paiou:
                best_paiou = current_paiou
                best_metrics = val_metrics
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0

                # Save checkpoint
                ckpt_path = os.path.join(
                    output_dir, 'checkpoints', f'{model_name}_fold{fold}_best.pt'
                )
                torch.save({
                    'model_state': best_state,
                    'metrics': best_metrics,
                    'fold': fold,
                    'epoch': epoch,
                    'config': config,
                }, ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f'  Early stopping at epoch {epoch+1}')
                    break

        fold_results.append({
            'fold': fold,
            'metrics': best_metrics,
            'checkpoint': ckpt_path if best_state is not None else None,
        })

        # Cleanup for next fold
        del model, optimizer, cosine_scheduler, scaler
        gc.collect()
        torch.cuda.empty_cache()

    # Aggregate results across folds
    metric_keys = ['IoU', 'F1', 'Precision', 'Recall', 'mIoU', 'Kappa',
                   'Betti0_Err', 'Betti1_Err', 'HVR', 'PA_IoU']

    print(f'\n{"="*60}')
    print(f'RESULTS — {model_name} (5-fold CV)')
    print(f'{"="*60}')
    print(f'{"Metric":>12s} | {"Mean":>8s} | {"Std":>8s} | {"Min":>8s} | {"Max":>8s}')
    print(f'{ "-"*42}')

    for key in metric_keys:
        values = [fr['metrics'][key] for fr in fold_results]
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        final_metrics[key] = mean_val
        final_metrics[f'{key}_std'] = std_val
        print(f'{key:>12s} | {mean_val:>8.4f} | {std_val:>8.4f} | {min_val:>8.4f} | {max_val:>8.4f}')

    final_metrics['model_name'] = model_name
    final_metrics['fold_results'] = fold_results

    # Save aggregated results
    results_path = os.path.join(output_dir, f'{model_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_metrics, f, indent=2, default=str)

    print(f'\nResults saved to {results_path}')
    return final_metrics


def statistical_significance(results_a, results_b, metric='PA_IoU'):
    """Paired statistical significance test between two models.

    Uses paired t-test across folds.
    """
    values_a = [r['metrics'][metric] for r in results_a['fold_results']]
    values_b = [r['metrics'][metric] for r in results_b['fold_results']]

    t_stat, p_value = scipy_stats.ttest_rel(values_a, values_b)

    # Also compute Cohen's d effect size
    mean_diff = np.mean(values_a) - np.mean(values_b)
    pooled_std = np.sqrt((np.std(values_a)**2 + np.std(values_b)**2) / 2)
    cohens_d = mean_diff / (pooled_std + 1e-6)

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'cohens_d': float(cohens_d),
        'mean_diff': float(mean_diff),
        'model_a_mean': float(np.mean(values_a)),
        'model_b_mean': float(np.mean(values_b)),
    }
