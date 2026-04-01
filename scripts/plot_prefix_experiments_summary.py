"""Plot summary figures for prefix recovery and corruption experiments."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

FIGDIR = '/weka/home/tli104/context_engineering/figures'

# ============================================================
# Color palette
# ============================================================
C_BLUE = '#2563eb'
C_ORANGE = '#ea580c'
C_GREEN = '#16a34a'
C_RED = '#dc2626'
C_PURPLE = '#7c3aed'
C_GRAY = '#6b7280'
C_TEAL = '#0d9488'

# ============================================================
# Figure 1: Direct Recovery vs 2-Turn Revision
# ============================================================
def fig1_recovery_vs_revision():
    models = ['Qwen3-4B', 'Qwen3-30B-A3B', 'Gemini 3 Flash', 'DeepSeek V3.2', 'MiniMax M2.5']
    direct   = [0.0, 0.4, 0.4, 4.5, 4.5]
    revision = [2.3, None, 7.0, 10.4, 17.1]  # No 30B revision run

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    w = 0.35

    bars1 = ax.bar(x - w/2, direct, w, label='Direct prefix continuation', color=C_BLUE, edgecolor='white', linewidth=0.5)
    rev_vals = [v if v is not None else 0 for v in revision]
    rev_colors = [C_ORANGE if v is not None else 'none' for v in revision]
    bars2 = ax.bar(x + w/2, rev_vals, w, label='2-turn revision', color=rev_colors, edgecolor='white', linewidth=0.5)

    # Value labels
    for bar, v in zip(bars1, direct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, v in zip(bars2, revision):
        if v is not None:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, 0.3,
                    'N/A', ha='center', va='bottom', fontsize=8, color=C_GRAY)

    ax.set_ylabel('Recovery Rate (%)', fontsize=12)
    ax.set_title('Self-Correction from Own Incorrect Solutions (IMOBench)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(0, 22)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = f'{FIGDIR}/recovery_vs_revision.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f'Saved {path}')
    plt.close()


# ============================================================
# Figure 2: Prefix Length Ablation (incorrect prefix continuation)
# ============================================================
def fig2_prefix_ablation():
    lengths_4b  = [0, 50, 100, 200, 400, 800, 1000]
    acc_4b      = [9.5, 9.1, 8.8, 9.2, 8.6, 8.6, 8.5]

    lengths_30b = [0, 50, 100, 200, 400, 800, 1000, 2000, 3000, 4000]
    acc_30b     = [13.6, 12.5, 13.0, 12.6, 13.0, 12.9, 11.2, 9.9, 9.7, 8.0]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lengths_4b, acc_4b, 'o-', color=C_BLUE, linewidth=2, markersize=7, label='Qwen3-4B-Instruct-2507')
    ax.plot(lengths_30b, acc_30b, 's-', color=C_ORANGE, linewidth=2, markersize=7, label='Qwen3-30B-A3B-Instruct-2507')

    # Horizontal baselines (fresh generation = p0)
    ax.axhline(9.5, color=C_BLUE, linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(13.6, color=C_ORANGE, linestyle='--', alpha=0.3, linewidth=1)

    ax.set_xlabel('Incorrect Prefix Length (tokens)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Continuation Accuracy After Incorrect Prefix (IMOBench, 16 samples/problem)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(5, 18)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = f'{FIGDIR}/prefix_length_ablation.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f'Saved {path}')
    plt.close()


# ============================================================
# Figure 3: Corruption Effect — Accuracy Drop from Clean Baseline
# ============================================================
def fig3_corruption_effect():
    # Data: clean accuracy, then corruption conditions
    # Format: (prefix_len, strategy, num_corrupt) -> accuracy
    data_4b = {
        'clean': {200: 80.4, 400: 80.7, 800: 81.0},
        ('rp', 1): {200: 79.8, 400: 79.8, 800: 80.7},
        ('rp', 3): {200: 79.7, 400: 80.2, 800: 80.3},
        ('rp', 5): {200: 79.2, 400: 78.5, 800: 78.9},
        ('ns', 1): {200: 79.9, 400: 79.4, 800: 79.7},
        ('ns', 3): {200: 80.0, 400: 78.1, 800: 79.0},
        ('ns', 5): {200: 79.6, 400: 79.3, 800: 78.2},
    }
    data_30b = {
        'clean': {200: 85.5, 400: 85.4, 800: 86.2},
        ('rp', 1): {200: 85.5, 400: 83.7, 800: 85.8},
        ('rp', 3): {200: 82.1, 400: 84.8, 800: 85.4},
        ('rp', 5): {200: 82.4, 400: 83.0, 800: 84.9},
        ('ns', 1): {200: 84.7, 400: 85.2, 800: 85.5},
        ('ns', 3): {200: 84.7, 400: 84.6, 800: 84.3},
        ('ns', 5): {200: 84.6, 400: 84.4, 800: 85.0},
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, data, title in [(axes[0], data_4b, 'Qwen3-4B-Instruct-2507'),
                             (axes[1], data_30b, 'Qwen3-30B-A3B-Instruct-2507')]:
        prefixes = [200, 400, 800]
        conditions = ['Clean', 'RP c=1', 'RP c=3', 'RP c=5', 'NS c=1', 'NS c=3', 'NS c=5']
        keys = ['clean', ('rp',1), ('rp',3), ('rp',5), ('ns',1), ('ns',3), ('ns',5)]
        colors = [C_GREEN, '#93c5fd', C_BLUE, '#1e3a5f', '#fdba74', C_ORANGE, '#9a3412']

        x = np.arange(len(prefixes))
        n = len(conditions)
        w = 0.11
        offsets = np.linspace(-(n-1)*w/2, (n-1)*w/2, n)

        for i, (key, cond, color) in enumerate(zip(keys, conditions, colors)):
            vals = [data[key][p] for p in prefixes]
            bars = ax.bar(x + offsets[i], vals, w, label=cond, color=color, edgecolor='white', linewidth=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels([f'p={p}' for p in prefixes], fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_ylim(75, 90)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].legend(fontsize=8, loc='lower left', ncol=2)

    fig.suptitle('Correct-Prefix Continuation: Clean vs. Corrupted (IMOBench, 16 samples)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = f'{FIGDIR}/corruption_grouped_bars.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f'Saved {path}')
    plt.close()


# ============================================================
# Figure 4: Dose-Response (accuracy vs num_corruptions)
# ============================================================
def fig4_dose_response():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    # Qwen3-4B
    rp_4b = {
        200: [80.4, 79.8, 79.7, 79.2],
        400: [80.7, 79.8, 80.2, 78.5],
        800: [81.0, 80.7, 80.3, 78.9],
    }
    ns_4b = {
        200: [80.4, 79.9, 80.0, 79.6],
        400: [80.7, 79.4, 78.1, 79.3],
        800: [81.0, 79.7, 79.0, 78.2],
    }

    # Qwen3-30B
    rp_30b = {
        200: [85.5, 85.5, 82.1, 82.4],
        400: [85.4, 83.7, 84.8, 83.0],
        800: [86.2, 85.8, 85.4, 84.9],
    }
    ns_30b = {
        200: [85.5, 84.7, 84.7, 84.6],
        400: [85.4, 85.2, 84.6, 84.4],
        800: [86.2, 85.5, 84.3, 85.0],
    }

    x_vals = [0, 1, 3, 5]
    colors_p = {200: C_BLUE, 400: C_ORANGE, 800: C_GREEN}
    markers = {200: 'o', 400: 's', 800: '^'}

    for ax, rp, ns, title in [(axes[0], rp_4b, ns_4b, 'Qwen3-4B-Instruct-2507'),
                                (axes[1], rp_30b, ns_30b, 'Qwen3-30B-A3B-Instruct-2507')]:
        for p in [200, 400, 800]:
            ax.plot(x_vals, rp[p], f'{markers[p]}-', color=colors_p[p], linewidth=2,
                    markersize=8, label=f'p={p} (result perturb.)')
            ax.plot(x_vals, ns[p], f'{markers[p]}--', color=colors_p[p], linewidth=1.5,
                    markersize=6, alpha=0.6, label=f'p={p} (number swap)')

        ax.set_xlabel('Number of Corruptions', fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x_vals)
        ax.set_xticklabels(['0\n(clean)', '1', '3', '5'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_ylim(76, 90)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].legend(fontsize=7.5, loc='lower left', ncol=2)

    fig.suptitle('Dose-Response: Accuracy vs. Number of Arithmetic Corruptions', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = f'{FIGDIR}/corruption_dose_response.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f'Saved {path}')
    plt.close()


# ============================================================
# Figure 5: Accuracy Drop Heatmap (clean - corrupted)
# ============================================================
def fig5_drop_heatmap():
    # Compute accuracy drop (clean - corrupted) for each condition
    # Rows: conditions (rp_c1..c5, ns_c1..c5), Cols: model x prefix_length
    conditions = ['RP c=1', 'RP c=3', 'RP c=5', 'NS c=1', 'NS c=3', 'NS c=5']
    col_labels = ['4B p200', '4B p400', '4B p800', '30B p200', '30B p400', '30B p800']

    clean_4b = [80.4, 80.7, 81.0]
    clean_30b = [85.5, 85.4, 86.2]

    corrupted = {
        'RP c=1': [79.8, 79.8, 80.7, 85.5, 83.7, 85.8],
        'RP c=3': [79.7, 80.2, 80.3, 82.1, 84.8, 85.4],
        'RP c=5': [79.2, 78.5, 78.9, 82.4, 83.0, 84.9],
        'NS c=1': [79.9, 79.4, 79.7, 84.7, 85.2, 85.5],
        'NS c=3': [80.0, 78.1, 79.0, 84.7, 84.6, 84.3],
        'NS c=5': [79.6, 79.3, 78.2, 84.6, 84.4, 85.0],
    }

    clean_all = clean_4b + clean_30b
    drops = np.array([[clean_all[j] - corrupted[c][j] for j in range(6)] for c in conditions])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(drops, cmap='YlOrRd', aspect='auto', vmin=0, vmax=4)

    # Annotate cells
    for i in range(len(conditions)):
        for j in range(6):
            val = drops[i, j]
            color = 'white' if val > 2.5 else 'black'
            ax.text(j, i, f'{val:+.1f}', ha='center', va='center', fontsize=10, fontweight='bold', color=color)

    ax.set_xticks(range(6))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions, fontsize=10)
    ax.set_title('Accuracy Drop from Clean Correct Prefix (pp)', fontsize=13, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Accuracy Drop (pp)', fontsize=10)

    plt.tight_layout()
    path = f'{FIGDIR}/corruption_drop_heatmap.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f'Saved {path}')
    plt.close()


if __name__ == '__main__':
    fig1_recovery_vs_revision()
    fig2_prefix_ablation()
    fig3_corruption_effect()
    fig4_dose_response()
    fig5_drop_heatmap()
    print('\nAll figures saved.')
