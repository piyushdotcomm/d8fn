"""Publication-quality visualization and results export."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch


def plot_results(all_results, output_dir, val_dataset=None):
    """Generate all figures for the paper.

    Args:
        all_results: dict of {model_name: metrics_dict}
        output_dir: Output directory.
        val_dataset: Optional dataset for qualitative samples.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style('whitegrid')
    colors = sns.color_palette('husl', n_colors=len(all_results))

    # 1. Bar chart comparison
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    metrics_plot = ['IoU', 'F1', 'Precision', 'Recall', 'mIoU',
                    'Kappa', 'PA_IoU', 'HVR', 'Betti0_Err', 'Betti1_Err']
    better_lower = {'HVR', 'Betti0_Err', 'Betti1_Err'}

    for ax, metric in zip(axes.flat, metrics_plot):
        names = list(all_results.keys())
        values = [all_results[n].get(metric, 0) for n in names]
        stds = [all_results[n].get(f'{metric}_std', 0) for n in names]

        bars = ax.bar(range(len(names)), values, yerr=stds, capsize=5, color=colors)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=7)
        ax.axhline(y=0, color='gray', linewidth=0.5)

        # Highlight best
        if metric in better_lower:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_metrics_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # 2. Radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    radar_metrics = ['IoU', 'F1', 'PA_IoU', 'mIoU', 'Kappa']
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    for idx, (name, results) in enumerate(all_results.items()):
        values = [results.get(m, 0) for m in radar_metrics]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_radar.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # 3. Results table
    table_data = []
    for name, results in all_results.items():
        row = {'Model': name}
        for m in ['IoU', 'F1', 'Precision', 'Recall', 'mIoU', 'Kappa', 'PA_IoU', 'HVR', 'Betti0_Err', 'Betti1_Err']:
            val = results.get(m, 0)
            std = results.get(f'{m}_std', 0)
            row[m] = f'{val:.4f} +/- {std:.4f}'
        table_data.append(row)

    df = pd.DataFrame(table_data)
    df.to_csv(os.path.join(output_dir, 'results_table.csv'), index=False)

    # Save as image
    fig, ax = plt.subplots(figsize=(14, len(df) * 0.5 + 1))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    plt.savefig(os.path.join(output_dir, 'fig3_results_table.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # 4. Qualitative predictions
    if val_dataset is not None and len(val_dataset) > 0:
        n_samples = min(5, len(val_dataset))
        flooded_indices = []
        for i in range(len(val_dataset)):
            if len(flooded_indices) >= n_samples:
                break
            _, _, _, _, _, _, _, mask, _ = val_dataset[i]
            if mask.sum() > 500:
                flooded_indices.append(i)

        if flooded_indices:
            n_rows = len(flooded_indices)
            n_models = len(all_results)
            fig, axes = plt.subplots(n_rows, n_models + 1, figsize=(4 * (n_models + 1), 4 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)

            for row, idx in enumerate(flooded_indices):
                item = val_dataset[idx]
                sar, dem, hand, slope, dem_raw, flow_dir, flow_acc, mask, raw_hand = item
                gt = mask[0].numpy()

                axes[row, 0].imshow(gt, cmap='Blues', vmin=0, vmax=1)
                axes[row, 0].set_title('Ground Truth' if row == 0 else '', fontsize=10)
                axes[row, 0].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'fig4_qualitative.png'), dpi=200, bbox_inches='tight')
            plt.close()

    # 5. Loss curves (if available)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, results in all_results.items():
        history = results.get('history', [])
        if history:
            epochs = range(1, len(history) + 1)
            losses = [h.get('val_loss', 0) for h in history]
            ax.plot(epochs, losses, 'o-', label=name, color=colors[list(all_results.keys()).index(name)])

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Validation Loss Curves', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_loss_curves.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print(f'All figures saved to {output_dir}/')
