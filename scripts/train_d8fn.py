"""Train D8FN using the full pipeline: D8FNLoss, EMA, TTA, HeightFieldHead.

This is the canonical training entry point that produces the paper's
reported results (IoU ~0.73, mIoU ~0.82 on KuroSiwo 5-fold CV).

Usage:
    python train_d8fn.py                          # Full D8FN, 5-fold CV
    python train_d8fn.py --model D8FN_Light       # Light variant
    python train_d8fn.py --model D8FN_NoRouting   # Ablation (no routing)
    python train_d8fn.py --model D8FN --fold 0    # Single fold
    python train_d8fn.py --no-height-field         # BCE+Dice only (no physics)
"""

import os, sys, argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch
from d8fn.models import MODEL_REGISTRY
from d8fn.train import run_5fold_cv
from d8fn.visualize import plot_results
from d8fn.data import ensure_data


def main():
    parser = argparse.ArgumentParser(description="Train D8FN")
    parser.add_argument("--model", default="D8FN",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model to train (default: D8FN)")
    parser.add_argument("--data-dir", default=os.path.join(ROOT, "data", "processed_v3"),
                        help="Path to .pt data files")
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "results", "d8fn"),
                        help="Output directory for checkpoints and results")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Base learning rate")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Maximum epochs per fold")
    parser.add_argument("--fold", type=int, default=None,
                        help="Run single fold (0-4) instead of 5-fold CV")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience")
    parser.add_argument("--no-height-field", action="store_true",
                        help="Use BCEDiceLoss instead of D8FNLoss (ablation)")
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable test-time augmentation")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"D8FN Training — {args.model}")
    print(f"{'='*60}")
    print(f"Data:     {args.data_dir}")
    print(f"Output:   {args.output_dir}")
    print(f"Batch:    {args.batch_size}")
    print(f"LR:       {args.lr}")
    print(f"Epochs:   {args.epochs}")
    print(f"TTA:      {not args.no_tta}")
    print(f"Height:   {not args.no_height_field}")
    print(f"Device:   {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()

    # Ensure data exists (auto-preprocess from tortilla if needed)
    if not ensure_data(args.data_dir):
        print(f"ERROR: Data not found at {args.data_dir}")
        print("Place Kuro Siwo .tortilla in Kaggle input or data/")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "in_ch": 9,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "patience": args.patience,
        "is_height_field": not args.no_height_field,
        "tta": not args.no_tta,
    }

    model_cls = MODEL_REGISTRY[args.model]

    if args.fold is not None:
        # Single fold
        from d8fn.train import train_epoch, evaluate, EMA
        from d8fn.losses import D8FNLoss, BCEDiceLoss
        from d8fn.data import create_dataloaders
        import copy, gc

        train_loader, val_loader = create_dataloaders(
            args.data_dir, batch_size=args.batch_size, fold=args.fold
        )

        model = model_cls(in_ch=9).to(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        is_height = config["is_height_field"]
        criterion = D8FNLoss() if is_height else BCEDiceLoss()

        encoder_params = []
        head_params = []
        for name, param in model.named_parameters():
            if "encoder" in name:
                encoder_params.append(param)
            else:
                head_params.append(param)

        enc_lr = args.lr * 0.3
        head_lr = args.lr
        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": enc_lr * 3.0},
            {"params": head_params, "lr": head_lr},
        ], weight_decay=1e-4)

        warmup_epochs = 5
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6
        )
        scaler = torch.amp.GradScaler()
        ema = EMA(model, decay=0.999)

        best_paiou = 0.0
        best_state = None
        patience_counter = 0

        for epoch in range(args.epochs):
            if epoch < warmup_epochs:
                scale = (epoch + 1) / warmup_epochs
                optimizer.param_groups[0]["lr"] = enc_lr * 3.0 * scale
                optimizer.param_groups[1]["lr"] = head_lr * scale
            else:
                cosine_scheduler.step()

            train_loss, train_comps = train_epoch(
                model, train_loader, criterion, optimizer, scaler, device, config, epoch
            )

            ema.update(model)

            ema_orig = {}
            for n, p in model.named_parameters():
                if p.requires_grad and n in ema.shadow:
                    ema_orig[n] = p.data.clone()
                    p.data = ema.shadow[n].clone()

            val_metrics = evaluate(model, val_loader, criterion, device, config)

            for n, p in model.named_parameters():
                if p.requires_grad and n in ema_orig:
                    p.data = ema_orig[n]

            paiou = val_metrics.get("PA_IoU", 0.0)
            lr_now = optimizer.param_groups[1]["lr"]
            marker = " *" if paiou > best_paiou else ""
            print(
                f"  Ep{epoch+1:02d} | LR:{lr_now:.2e} | "
                f"Loss:{train_loss:.4f} | IoU:{val_metrics['IoU']:.4f} | "
                f"F1:{val_metrics['F1']:.4f} | mIoU:{val_metrics['mIoU']:.4f} | "
                f"PA-IoU:{paiou:.4f}{marker}"
            )

            if paiou > best_paiou:
                best_paiou = paiou
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                ckpt_path = os.path.join(
                    args.output_dir, "checkpoints",
                    f"{args.model}_fold{args.fold}_best.pt"
                )
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save({
                    "model_state": best_state,
                    "metrics": val_metrics,
                    "fold": args.fold,
                    "epoch": epoch,
                    "config": config,
                }, ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stop E{epoch+1}")
                    break

        print(f"\nFold {args.fold} done — best PA-IoU: {best_paiou:.4f}")
    else:
        # 5-fold CV
        results = run_5fold_cv(
            model_class=model_cls,
            model_name=args.model,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config=config,
            num_folds=5,
            device=device,
        )

        model_name = args.model
        plot_results({model_name: results}, args.output_dir)

        print(f"\nDone — results in {args.output_dir}/")


if __name__ == "__main__":
    main()
