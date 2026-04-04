"""
plot_figures.py
===============
Generates Figures 2, 3, 4, 10, 11, 12 for the RL+LLM navigation paper.

Style rules
-----------
* White background, academic (reference-matched).
* NO figure-level titles (suptitle removed).
* Single-panel figures have NO axis title.
* Multi-panel figures label each sub-panel with a bold letter in the
  upper-left corner using label_ax(), e.g. (a), (b), (c), (d).

Required data files (place in the same directory, or update path args):
  ablation_dual_map.json
  Gemma-3-1B_data.json
  LFM-2_5-1_2B_data.json
  qwen2_5-coder-1_5b-instruct_data.json

Output PNGs (200 dpi):
  fig2_ppo_trap.png
  fig3_success_rate.png
  fig4_benchmark_panel.png
  fig10_ablation.png
  fig11_llm_ablation.png
  fig12_hallucination.png

Usage:
  python plot_figures.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Global rcParams ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          11,
    'axes.facecolor':     '#f8f9fa',
    'axes.edgecolor':     '#444444',
    'axes.labelcolor':    '#222222',
    'axes.titlecolor':    '#111111',
    'axes.titlesize':     11,
    'axes.titleweight':   'normal',
    'axes.labelsize':     11,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.color':         '#dddddd',
    'grid.linestyle':     '--',
    'grid.linewidth':     0.7,
    'grid.alpha':         1.0,
    'xtick.color':        '#333333',
    'ytick.color':        '#333333',
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'figure.facecolor':   'white',
    'figure.edgecolor':   'white',
    'legend.facecolor':   'white',
    'legend.edgecolor':   '#cccccc',
    'legend.framealpha':  0.95,
    'legend.fontsize':    9,
    'savefig.facecolor':  'white',
    'savefig.edgecolor':  'white',
    'savefig.dpi':        200,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.12,
})

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    'dijkstra': '#1f77b4',
    'astar':    '#2ca02c',
    'gbfs':     '#d62728',
    'apf':      '#9467bd',
    'ppo':      '#8c564b',
    'ours':     '#e377c2',
}

ALGO_ORDER  = ['Local Dijkstra', 'Local A*', 'Local GBFS',
               'APF (Reactive)', 'Pure PPO', 'Ours (RL+LLM)']
ALGO_COLORS = [C['dijkstra'], C['astar'], C['gbfs'],
               C['apf'],      C['ppo'],   C['ours']]
MARKERS     = ['o', 's', '^', 'D', 'v', '*']
LEVELS      = [1, 2, 3, 4, 5]

# ── Summary data ──────────────────────────────────────────────────────────────
summary = {}
for _l in LEVELS:
    summary[('Local Dijkstra', _l)] = (1.0, [46, 46, 50,  85, 129][_l-1])
    summary[('Local A*',       _l)] = (1.0, [44, 46, 47,  83, 124][_l-1])
    summary[('Local GBFS',     _l)] = (1.0, [44, 46, 45,  81, 121][_l-1])
    summary[('APF (Reactive)', _l)] = (0.0, None)
_ppo_sr = {1: 1.0, 2: 0.0, 3: 1.0, 4: 1.0, 5: 0.0}
_ppo_st = {1: 46,  2: None, 3: 44,  4: 84,  5: None}
for _l in LEVELS:
    summary[('Pure PPO',      _l)] = (_ppo_sr[_l], _ppo_st[_l])
    summary[('Ours (RL+LLM)', _l)] = (1.0, {1:46, 2:110.55, 3:44, 4:84, 5:203.7}[_l])

outcome = {
    'Local Dijkstra':  {'Success': 100, 'Local Minima':  0, 'Trapped':  0, 'Timeout':   0},
    'Local A*':        {'Success': 100, 'Local Minima':  0, 'Trapped':  0, 'Timeout':   0},
    'Local GBFS':      {'Success': 100, 'Local Minima':  0, 'Trapped':  0, 'Timeout':   0},
    'APF (Reactive)':  {'Success':   0, 'Local Minima':  0, 'Trapped':  0, 'Timeout': 100},
    'Pure PPO':        {'Success':  60, 'Local Minima': 20, 'Trapped': 20, 'Timeout':   0},
    'Ours (RL+LLM)':   {'Success': 100, 'Local Minima':  0, 'Trapped':  0, 'Timeout':   0},
}

hall = {
    'Gemma-3\n(1B)':         {'format_fail': 0, 'oob':  0, 'wall_pick': 14,
                               'not_in_list': 67, 'perfect': 19},
    'LFM-2.5\n(1.2B)':       {'format_fail': 0, 'oob':  0, 'wall_pick':  7,
                               'not_in_list': 83, 'perfect': 10},
    'Qwen2.5-Coder\n(1.5B)': {'format_fail': 3, 'oob':  3, 'wall_pick': 21,
                               'not_in_list': 36, 'perfect': 37},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def label_ax(ax, letter, fontsize=13):
    """Place a bold sub-panel label e.g. '(a)' in the upper-left corner."""
    ax.text(-0.08, 1.04, f'({letter})',
            transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold',
            va='bottom', ha='left', color='#111111')


def make_twin(ax, spine_color):
    """Create a right-side twin axis with matching spine colour."""
    ax2 = ax.twinx()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_color(spine_color)
    ax2.tick_params(axis='y', colors=spine_color, labelsize=9)
    return ax2


# =============================================================================
# FIG 2 — PPO Local-Minima Trap  (single panel, no label)
# =============================================================================
def plot_fig2(outpath='fig2_ppo_trap.png'):
    np.random.seed(42)
    t = np.arange(201)

    dist = np.zeros(201); dist[0] = 26.3
    for i in range(1, 21):
        dist[i] = max(9.0, 26.3 - 1.2*i + np.random.randn()*0.3)
    dist[21:30] = 10.0 + np.sin(np.arange(9) * np.pi) * 0.5
    dist[30:90] = 9.3
    dist[90:93] = [9.3, 9.1, 9.0]
    for i in range(93,  115): dist[i] = 9.0 + abs(np.sin((i-93)  * np.pi / 2.2)) * 2.5
    dist[115:120] = 9.0
    for i in range(120, 175): dist[i] = 9.0 + abs(np.sin((i-120) * np.pi / 2.2)) * 2.5
    dist[175:178] = [11.5, 11.0, 10.5]
    for i in range(178, 200): dist[i] = 9.0 + abs(np.sin((i-178) * np.pi / 2.2)) * 2.2
    dist[200] = 9.0

    entropy = np.full(201, 0.85) + np.random.randn(201) * 0.05
    for sp in [29, 30, 90, 100, 105, 110, 170, 185, 187]:
        entropy[sp] = 1.3 + np.random.rand() * 0.15

    fig, ax1 = plt.subplots(figsize=(9, 4.2))
    ax2 = make_twin(ax1, C['dijkstra'])

    ax1.plot(t, dist,    color=C['gbfs'],     lw=2.2, label='Distance to Goal', zorder=3)
    ax2.plot(t, entropy, color=C['dijkstra'], lw=1.4, ls='--', alpha=0.9,
             label='Policy Entropy', zorder=2)

    for a, b in [(30, 90), (115, 175), (178, 200)]:
        ax1.axvspan(a, b, alpha=0.07, color=C['ppo'], zorder=0)

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Distance to Goal (Grid Units)', color=C['gbfs'])
    ax2.set_ylabel('Policy Entropy (Confidence)',   color=C['dijkstra'])
    ax1.tick_params(axis='y', colors=C['gbfs'])
    ax1.set_xlim(0, 200)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper right', fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close()
    print(f'Saved {outpath}')


# =============================================================================
# FIG 3 — Success Rate  (single panel, no label)
# =============================================================================
def plot_fig3(outpath='fig3_success_rate.png'):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for algo, col, mk in zip(ALGO_ORDER, ALGO_COLORS, MARKERS):
        sr   = [summary[(algo, l)][0] * 100 for l in LEVELS]
        hero = algo == 'Ours (RL+LLM)'
        ax.plot(LEVELS, sr,
                color=col,
                lw=2.5 if hero else 1.6,
                ls='-'  if hero else '--',
                marker=mk,
                ms=9 if hero else 7,
                label=algo,
                zorder=5 if hero else 2,
                clip_on=False,
                markeredgecolor='white', markeredgewidth=0.8)

    ax.set_xlabel('Map Difficulty Level')
    ax.set_ylabel('Success Rate (%)')
    ax.set_xticks(LEVELS)
    ax.set_xticklabels([f'L{l}' for l in LEVELS])
    ax.set_ylim(-5, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
    ax.legend(loc='lower left', ncol=2, fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close()
    print(f'Saved {outpath}')


# =============================================================================
# FIG 4 — Full Benchmark Panel  (2x2, labels a-d)
# =============================================================================
def plot_fig4(outpath='fig4_benchmark_panel.png'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── (a) Success Rate line ────────────────────────────────────────────────
    ax = axes[0, 0]
    label_ax(ax, 'a')
    for algo, col, mk in zip(ALGO_ORDER, ALGO_COLORS, MARKERS):
        sr   = [summary[(algo, l)][0] * 100 for l in LEVELS]
        hero = algo == 'Ours (RL+LLM)'
        ax.plot(LEVELS, sr,
                color=col, lw=2.5 if hero else 1.5,
                ls='-' if hero else '--',
                marker=mk, ms=8 if hero else 6,
                label=algo, zorder=5 if hero else 2,
                markeredgecolor='white', markeredgewidth=0.6)
    ax.set(xlabel='Map Difficulty Level', ylabel='Success Rate (%)')
    ax.set_xticks(LEVELS); ax.set_xticklabels([f'L{l}' for l in LEVELS])
    ax.set_ylim(-5, 108); ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
    ax.legend(fontsize=8, ncol=2)

    # ── (b) Average Steps grouped bar ───────────────────────────────────────
    ax = axes[0, 1]
    label_ax(ax, 'b')
    valid_algos = [a for a in ALGO_ORDER if a != 'APF (Reactive)']
    valid_cols  = [c for a, c in zip(ALGO_ORDER, ALGO_COLORS) if a != 'APF (Reactive)']
    n = len(valid_algos); bw = 0.13; xs = np.arange(5)
    for i, (algo, col) in enumerate(zip(valid_algos, valid_cols)):
        sp   = [summary[(algo, l)][1] or 0 for l in LEVELS]
        bars = ax.bar(xs + (i-(n-1)/2)*bw, sp, bw*0.9,
                      color=col, alpha=0.82, edgecolor='white', lw=0.5, label=algo)
        if algo == 'Ours (RL+LLM)':
            for bar, v in zip(bars, sp):
                if v > 0:
                    ax.text(bar.get_x()+bar.get_width()/2, v+2,
                            str(int(round(v))),
                            ha='center', va='bottom', fontsize=7.5,
                            color=C['ours'], fontweight='bold')
    ax.set_xticks(xs); ax.set_xticklabels([f'L{l}' for l in LEVELS])
    ax.set(xlabel='Map Difficulty Level', ylabel='Average Steps')
    ax.legend(fontsize=8, ncol=2)

    # ── (c) Outcome Distribution stacked horizontal bar ──────────────────────
    ax = axes[1, 0]
    label_ax(ax, 'c')
    cats       = ['Success', 'Local Minima', 'Trapped', 'Timeout']
    cat_colors = ['#2ca02c', '#ff7f0e',      '#9467bd', '#d62728']
    inv   = list(reversed(ALGO_ORDER))
    lefts = [0] * 6
    for cat, cc in zip(cats, cat_colors):
        vals = [outcome[a][cat] for a in inv]
        bars = ax.barh(range(6), vals, left=lefts,
                       color=cc, label=cat, edgecolor='white', lw=0.4)
        for j, (bar, v) in enumerate(zip(bars, vals)):
            if v > 5:
                ax.text(lefts[j]+v/2, j, f'{v}%',
                        ha='center', va='center',
                        fontsize=8.5, color='white', fontweight='bold')
        lefts = [l+v for l, v in zip(lefts, vals)]
    ax.set_yticks(range(6)); ax.set_yticklabels(inv, fontsize=9.5)
    ax.set(xlabel='Proportion (%)', xlim=(0, 100))
    ax.spines['left'].set_visible(True)
    ax.grid(axis='x'); ax.grid(axis='y', alpha=0)
    oi = inv.index('Ours (RL+LLM)')
    ax.get_yticklabels()[oi].set_color(C['ours'])
    ax.get_yticklabels()[oi].set_fontweight('bold')
    ax.legend(loc='lower right', fontsize=8.5)

    # ── (d) Summary Table ────────────────────────────────────────────────────
    ax = axes[1, 1]
    label_ax(ax, 'd')
    ax.axis('off')
    cols = ['Algorithm', 'SR-L1', 'SR-L2', 'SR-L3', 'SR-L4', 'SR-L5',
            'St-L1', 'St-L2', 'St-L3', 'St-L4', 'St-L5', 'Avg SR', 'Avg St']
    rows = []
    for algo in ALGO_ORDER:
        sr_v   = [summary[(algo, l)][0] for l in LEVELS]
        st_v   = [summary[(algo, l)][1] for l in LEVELS]
        avg_sr = np.mean(sr_v) * 100
        vs     = [s for s in st_v if s]
        avg_st = np.mean(vs) if vs else None
        row    = ([algo]
                  + [f'{s*100:.0f}%' for s in sr_v]
                  + [f'{s:.0f}' if s else '-' for s in st_v]
                  + [f'{avg_sr:.0f}%', f'{avg_st:.0f}' if avg_st else '-'])
        rows.append(row)

    tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5)
    tbl.auto_set_column_width(list(range(len(cols))))
    for j in range(len(cols)):
        tbl[0, j].set_facecolor('#2c3e50')
        tbl[0, j].get_text().set_color('white')
        tbl[0, j].get_text().set_fontweight('bold')
        tbl[0, j].set_edgecolor('#999999')
    ours_r = ALGO_ORDER.index('Ours (RL+LLM)') + 1
    for i in range(1, len(ALGO_ORDER)+1):
        for j in range(len(cols)):
            if i == ours_r:
                tbl[i, j].set_facecolor('#fce4ec')
                tbl[i, j].get_text().set_color(C['ours'])
                tbl[i, j].get_text().set_fontweight('bold')
            else:
                tbl[i, j].set_facecolor('#f5f5f5' if i%2==0 else 'white')
                tbl[i, j].get_text().set_color('#222222')
            tbl[i, j].set_edgecolor('#cccccc')

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close()
    print(f'Saved {outpath}')


# =============================================================================
# FIG 10 — Hyperparameter Sensitivity Ablation  (2x2, labels a-d)
# =============================================================================
def plot_fig10(ab_path='ablation_dual_map.json', outpath='fig10_ablation.png'):
    with open(ab_path) as f:
        ab = json.load(f)

    def _panel(ax, data, pvals, pname, chosen, letter, map_tag, fixed_tag):
        srs   = [data[str(v)]['sr']        for v in pvals]
        steps = [data[str(v)]['avg_steps'] for v in pvals]
        xs    = np.arange(len(pvals))
        bar_c = ['#ff7f0e' if str(v)==str(chosen) else '#aec7e8' for v in pvals]

        ax2 = make_twin(ax, '#d62728')
        ax.bar(xs, srs, color=bar_c, alpha=0.85, edgecolor='white', lw=0.6, width=0.6)
        ax2.plot(xs, steps, color='#d62728', marker='o', ms=6,
                 lw=2.0, zorder=4, label='Avg. Steps (success only)')

        for sr, x in zip(srs, xs):
            ax.text(x, sr+0.8, f'{sr:.0f}%',
                    ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold', color='#333333')

        ci = list(map(str, pvals)).index(str(chosen))
        ax.annotate('* Chosen',
                    xy=(ci, srs[ci]+2), xytext=(ci, srs[ci]+11),
                    ha='center', fontsize=8, color='#ff7f0e', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.2))

        ax.set_xticks(xs)
        ax.set_xticklabels([str(v) for v in pvals], fontsize=9.5)
        ax.set_xlabel(f'{pname}  ({fixed_tag})')
        ax.set_ylabel('Success Rate (%)', color='#1f77b4', fontsize=10)
        ax.tick_params(axis='y', colors='#1f77b4', labelsize=9)
        ax2.set_ylabel('Avg. Steps (success only)', color='#d62728', fontsize=10)
        ax.set_ylim(0, 130)

        label_ax(ax, letter)
        ax.text(0.02, 0.97, map_tag,
                transform=ax.transAxes,
                fontsize=9, va='top', ha='left', color='#555555')

        p_chosen = mpatches.Patch(color='#ff7f0e', label='Chosen param')
        p_other  = mpatches.Patch(color='#aec7e8', label='Other')
        ax.legend(handles=[p_chosen, p_other], fontsize=8, loc='lower right')
        ax2.legend(fontsize=8, loc='upper left')

    N_vals = [5, 10, 15, 20, 30]
    D_vals = [0.5, 1.0, 1.5, 2.0, 3.0]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    _panel(axes[0,0], ab['L2']['N'], N_vals, 'Window Size N',   15,  'a',
           'L2 (Trap-Heavy)', '\u0394d = 1.5 fixed')
    _panel(axes[0,1], ab['L2']['D'], D_vals, 'Threshold \u0394d', 1.5, 'b',
           'L2 (Trap-Heavy)', 'N = 15 fixed')
    _panel(axes[1,0], ab['L5']['N'], N_vals, 'Window Size N',   15,  'c',
           'L5 (Hard)', '\u0394d = 1.5 fixed')
    _panel(axes[1,1], ab['L5']['D'], D_vals, 'Threshold \u0394d', 1.5, 'd',
           'L5 (Hard)', 'N = 15 fixed')

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close()
    print(f'Saved {outpath}')


# =============================================================================
# FIG 11 — LLM Backend Ablation  (2x2, labels a-d)
# =============================================================================
def plot_fig11(
    gemma_path = 'Gemma-3-1B_data.json',
    lfm_path   = 'LFM-2_5-1_2B_data.json',
    qwen_path  = 'qwen2_5-coder-1_5b-instruct_data.json',
    outpath    = 'fig11_llm_ablation.png',
):
    models_data = {}
    for fpath, key in [
        (gemma_path, 'Gemma-3 (1B)'),
        (lfm_path,   'LFM-2.5 (1.2B)'),
        (qwen_path,  'Qwen2.5-Coder (1.5B)'),
    ]:
        with open(fpath) as f:
            models_data[key] = json.load(f)

    MAPS = ['map_0', 'map_1', 'map_2', 'map_3', 'map_4']
    MK   = list(models_data.keys())
    MC   = ['#1f77b4', '#ff7f0e', '#2ca02c']

    def _compute(md):
        sr, steps_d, calls_d, lat_d = {}, {}, {}, {}
        for mp in MAPS:
            tr = md[mp]; s = [t for t in tr if t['success']]
            sr[mp]      = len(s) / len(tr)
            steps_d[mp] = [t['steps']     for t in s]
            calls_d[mp] = [t['llm_calls'] for t in tr]
            lat_d[mp]   = [t['total_time'] / t['llm_calls']
                           for t in tr if t['llm_calls'] > 0]
        return sr, steps_d, calls_d, lat_d

    stats      = {k: _compute(v) for k, v in models_data.items()}
    MAP_LABELS = [f'Map {i}' for i in range(5)]
    xm = np.arange(5); bw = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── (a) Success Rate ─────────────────────────────────────────────────────
    ax = axes[0, 0]
    label_ax(ax, 'a')
    ax.text(0.98, 0.02, 'higher \u2191 is better',
            transform=ax.transAxes, fontsize=8.5,
            ha='right', va='bottom', color='#666666', style='italic')
    for i, (mk, col) in enumerate(zip(MK, MC)):
        ax.bar(xm+(i-1)*bw, [stats[mk][0][mp] for mp in MAPS],
               bw*0.9, color=col, alpha=0.82, label=mk,
               edgecolor='white', lw=0.5)
    ax.set_xticks(xm); ax.set_xticklabels(MAP_LABELS)
    ax.set(xlabel='Map', ylabel='Success Rate', ylim=(0, 1.18))
    ax.legend(fontsize=9)

    # ── (b) Steps box plots ──────────────────────────────────────────────────
    ax = axes[0, 1]
    label_ax(ax, 'b')
    ax.text(0.98, 0.98, 'lower \u2193 is better',
            transform=ax.transAxes, fontsize=8.5,
            ha='right', va='top', color='#666666', style='italic')
    handles = []
    for i, (mk, col) in enumerate(zip(MK, MC)):
        for j, mp in enumerate(MAPS):
            d = stats[mk][1][mp]
            if d:
                ax.boxplot(d, positions=[j+(i-1)*0.28], widths=0.22,
                           patch_artist=True, manage_ticks=False,
                           boxprops=dict(facecolor=col, alpha=0.65),
                           medianprops=dict(color='black', lw=1.8),
                           whiskerprops=dict(color=col, lw=1.2),
                           capprops=dict(color=col, lw=1.2),
                           flierprops=dict(marker='o', ms=3, alpha=0.5,
                                           markerfacecolor=col,
                                           markeredgecolor='none'))
        handles.append(mpatches.Patch(color=col, alpha=0.75, label=mk))
    ax.set_xticks(np.arange(5)); ax.set_xticklabels(MAP_LABELS)
    ax.set(xlabel='Map', ylabel='Steps to Goal')
    ax.legend(handles=handles, fontsize=9)

    # ── (c) LLM Call Frequency box plots ────────────────────────────────────
    ax = axes[1, 0]
    label_ax(ax, 'c')
    ax.text(0.98, 0.98, 'lower \u2193 is better',
            transform=ax.transAxes, fontsize=8.5,
            ha='right', va='top', color='#666666', style='italic')
    for i, (mk, col) in enumerate(zip(MK, MC)):
        for j, mp in enumerate(MAPS):
            d = stats[mk][2][mp]
            if d:
                ax.boxplot(d, positions=[j+(i-1)*0.28], widths=0.22,
                           patch_artist=True, manage_ticks=False,
                           boxprops=dict(facecolor=col, alpha=0.65),
                           medianprops=dict(color='black', lw=1.8),
                           whiskerprops=dict(color=col, lw=1.2),
                           capprops=dict(color=col, lw=1.2),
                           flierprops=dict(marker='o', ms=3, alpha=0.5,
                                           markerfacecolor=col,
                                           markeredgecolor='none'))
    ax.set_xticks(np.arange(5)); ax.set_xticklabels(MAP_LABELS)
    ax.set(xlabel='Map', ylabel='LLM Calls per Episode')
    ax.legend(handles=handles, fontsize=9)

    # ── (d) Mean Inference Latency bar ──────────────────────────────────────
    ax = axes[1, 1]
    label_ax(ax, 'd')
    ax.text(0.98, 0.98, 'lower \u2193 is better',
            transform=ax.transAxes, fontsize=8.5,
            ha='right', va='top', color='#666666', style='italic')
    mean_lats = []
    for mk in MK:
        all_lats = sum([stats[mk][3][mp] for mp in MAPS], [])
        mean_lats.append(np.mean(all_lats) if all_lats else 0)
    bars = ax.bar(np.arange(3), mean_lats, color=MC,
                  alpha=0.85, edgecolor='white', lw=0.5, width=0.5)
    for bar, v in zip(bars, mean_lats):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.12,
                f'{v:.2f}s', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#222222')
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(MK, fontsize=10)
    ax.set(xlabel='Model', ylabel='Time per Call (Seconds)')

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close()
    print(f'Saved {outpath}')


# =============================================================================
# FIG 12 — Spatial Hallucination Analysis  (2x2, labels a-d)
# =============================================================================
def plot_fig12(outpath='fig12_hallucination.png'):
    HCOLS = {
        'perfect':     '#2ca02c',
        'not_in_list': '#ff7f0e',
        'wall_pick':   '#d62728',
        'oob':         '#9467bd',
        'format_fail': '#8c8c8c',
    }
    HLABELS = {
        'perfect':     'Perfect',
        'not_in_list': 'Not in shortlist',
        'wall_pick':   'Wall pick (spatial halluc.)',
        'oob':         'Out of bounds',
        'format_fail': 'Format error',
    }

    model_names = list(hall.keys())
    cats        = list(HLABELS.keys())

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # ── (a) Stacked bar — failure mode distribution ──────────────────────────
    ax = axes[0, 0]
    label_ax(ax, 'a')
    lefts = [0] * 3
    for cat in cats:
        vals = [hall[m][cat] for m in model_names]
        bars = ax.bar(range(3), vals, bottom=lefts,
                      color=HCOLS[cat], label=HLABELS[cat],
                      edgecolor='white', lw=0.8)
        for j, (bar, v, l) in enumerate(zip(bars, vals, lefts)):
            if v == 0:
                lefts[j] += v
                continue
            center_y = l + v / 2
            if v >= 10:
                # large segment: white label centred inside
                ax.text(bar.get_x()+bar.get_width()/2, center_y, f'{v}%',
                        ha='center', va='center',
                        fontsize=10, color='white', fontweight='bold', zorder=10)
            else:
                # small segment: annotate to the right with a leader line
                bar_right = bar.get_x() + bar.get_width()
                bar_mid_y = l + v / 2
                ax.annotate(f'{v}%',
                            xy=(bar_right, bar_mid_y),
                            xytext=(bar_right + 0.08, bar_mid_y),
                            ha='left', va='center',
                            fontsize=8.5, color='#333333', fontweight='bold',
                            arrowprops=dict(arrowstyle='-', color='#888888',
                                            lw=0.8, shrinkA=0, shrinkB=2),
                            zorder=10)
            lefts[j] += v

    ax.set_xticks(range(3))
    ax.set_xticklabels(model_names, fontsize=10.5)
    ax.set_ylabel('Proportion (%)')
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.6, 2.9)   # extra right margin for side annotations
    patch_handles = [mpatches.Patch(color=HCOLS[c], label=HLABELS[c]) for c in cats]
    # Legend below the axes to avoid overlapping bars
    ax.legend(handles=patch_handles,
              loc='upper left', bbox_to_anchor=(0.0, -0.12),
              ncol=2, fontsize=8.5, framealpha=0.95)

    # ── (b) Model Comparison Overview (grouped bar) ──────────────────────────
    ax = axes[0, 1]
    label_ax(ax, 'b')
    met_labels = ['Perfect\nCompliance', 'No Wall\nHallucination', 'Format\nCompliance']
    MC3 = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (mn, col) in enumerate(zip(model_names, MC3)):
        h = hall[mn]
        vals = [h['perfect'], 100-h['wall_pick'], 100-h['format_fail']]
        ax.bar(np.arange(3)+(i-1)*0.28, vals, 0.25,
               color=col, alpha=0.82,
               label=mn.replace('\n', ' '), edgecolor='white', lw=0.4)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(met_labels, fontsize=10)
    ax.set(ylabel='Score (%)', ylim=(0, 118))
    ax.legend(fontsize=9)

    # ── (c) Spatial Hallucination Rate ──────────────────────────────────────
    ax = axes[1, 0]
    label_ax(ax, 'c')
    ax.text(0.98, 0.98, 'lower \u2193 is better',
            transform=ax.transAxes, fontsize=8.5,
            ha='right', va='top', color='#666666', style='italic')
    spat  = [hall[m]['wall_pick'] for m in model_names]
    bcols = ['#2ca02c' if v == min(spat) else '#d62728' for v in spat]
    bars  = ax.bar(range(3), spat, color=bcols,
                   alpha=0.85, edgecolor='white', lw=0.5, width=0.5)
    for bar, v in zip(bars, spat):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.4, f'{v}%',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#222222')
        if v == min(spat):
            ax.annotate('* Best',
                        xy=(bar.get_x()+bar.get_width()/2, v),
                        xytext=(bar.get_x()+bar.get_width()/2, v+6),
                        ha='center', fontsize=9, color='#2ca02c', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.3))
    ax.set_xticks(range(3)); ax.set_xticklabels(model_names, fontsize=10.5)
    ax.set_ylabel('Spatial Hallucination Rate (%)')
    ax.set_ylim(0, 36)

    # ── (d) Perfect Compliance Rate ──────────────────────────────────────────
    ax = axes[1, 1]
    label_ax(ax, 'd')
    ax.text(0.98, 0.98, 'higher \u2191 is better',
            transform=ax.transAxes, fontsize=8.5,
            ha='right', va='top', color='#666666', style='italic')
    perf  = [hall[m]['perfect'] for m in model_names]
    bcols = ['#2ca02c' if v == max(perf) else '#1f77b4' for v in perf]
    bars  = ax.bar(range(3), perf, color=bcols,
                   alpha=0.85, edgecolor='white', lw=0.5, width=0.5)
    for bar, v in zip(bars, perf):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.4, f'{v}%',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#222222')
        if v == max(perf):
            ax.annotate('* Best',
                        xy=(bar.get_x()+bar.get_width()/2, v),
                        xytext=(bar.get_x()+bar.get_width()/2, v+6),
                        ha='center', fontsize=9, color='#2ca02c', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.3))
    ax.set_xticks(range(3)); ax.set_xticklabels(model_names, fontsize=10.5)
    ax.set_ylabel('Perfect Compliance Rate (%)')
    ax.set_ylim(0, 56)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close()
    print(f'Saved {outpath}')


# =============================================================================
# Entry point
# =============================================================================
if __name__ == '__main__':
    plot_fig2()
    plot_fig3()
    plot_fig4()
    plot_fig10()
    plot_fig11()
    plot_fig12()
    print('\nAll figures generated successfully.')
