# scripts/exp5_replot.py
# 直接读取已有的 ablation_dual_map.json，重新绘图
# 用法：python scripts/exp5_replot.py

import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(project_root, "results")

json_path = os.path.join(SAVE_DIR, "ablation_dual_map.json")
with open(json_path, "r", encoding="utf-8") as f:
    all_results = json.load(f)

print("✅ 数据加载成功，开始绘图...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.patch.set_facecolor('#fafafa')

map_labels = {"L2": "Level 2  (Trap-Heavy)", "L5": "Level 5  (Hard)"}
col_titles = [
    r"(a)  Sensitivity to Window Size $N$   ($\Delta d = 1.5$ fixed)",
    r"(b)  Sensitivity to Threshold $\Delta d$   ($N = 15$ fixed)",
]
bar_colors = ['#3498db', '#2ecc71']
line_color = '#e74c3c'
HIGHLIGHT  = {'N': '15', 'D': '1.5'}

for row, map_name in enumerate(["L2", "L5"]):
    for col, key in enumerate(["N", "D"]):
        ax  = axes[row][col]
        ax2 = ax.twinx()
        ax.set_facecolor('white')

        data  = all_results[map_name][key]
        ks    = list(data.keys())
        calls = [data[k]['avg_calls'] for k in ks]
        steps = [data[k]['avg_steps'] if data[k]['avg_steps'] is not None
                 and not (isinstance(data[k]['avg_steps'], float)
                          and data[k]['avg_steps'] != data[k]['avg_steps'])
                 else 0 for k in ks]
        srs   = [data[k]['sr'] for k in ks]

        hi     = HIGHLIGHT[key]
        colors = [('#FF8C00' if k == hi else bar_colors[col]) for k in ks]
        bars   = ax.bar(ks, calls, color=colors, alpha=0.82, width=0.5, zorder=2)

        # 顶部留足空间
        ax.set_ylim(0, max(calls) * 1.65)
        step_min = min(s for s in steps if s > 0) if any(s > 0 for s in steps) else 0
        ax2.set_ylim(step_min * 0.85, max(steps) * 1.18)

        ax.tick_params(axis='y', labelcolor=bar_colors[col], labelsize=8)
        ax2.tick_params(axis='y', labelcolor=line_color, labelsize=8)
        ax2.spines['top'].set_visible(False)

        ax2.plot(ks, steps, color=line_color, marker='o',
                 linewidth=2.5, markersize=7, zorder=3)

        # SR% 标注：紧贴柱顶，单行，不换行
        for bar, sr in zip(bars, srs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(calls) * 0.03,
                    f"{sr:.0f}%", ha='center', va='bottom',
                    fontsize=8.5, color='#222', fontweight='bold')

        # ★ Chosen：箭头向下，文字在上方留白区，不碰标题
        if hi in ks:
            i = ks.index(hi)
            bar_hi = bars[i]
            ax.annotate(
                '★ Chosen',
                xy=(bar_hi.get_x() + bar_hi.get_width() / 2,
                    bar_hi.get_height() + max(calls) * 0.05),
                xytext=(bar_hi.get_x() + bar_hi.get_width() / 2,
                        max(calls) * 1.48),
                arrowprops=dict(facecolor='#FF8C00', edgecolor='#FF8C00',
                                shrink=0.06, width=1.5, headwidth=6),
                ha='center', fontsize=8.5,
                color='#FF8C00', fontweight='bold',
            )

        # 列标题只在第一行显示，pad 留足
        if row == 0:
            ax.set_title(col_titles[col], fontsize=10,
                         fontweight='bold', pad=18)

        # 左侧行标签
        if col == 0:
            ax.set_ylabel(f"{map_labels[map_name]}\n\nAvg. LLM Calls / Ep",
                          color=bar_colors[col], fontsize=9, fontweight='bold')
        else:
            ax.set_ylabel('Avg. LLM Calls / Ep',
                          color=bar_colors[col], fontsize=9, fontweight='bold')

        ax2.set_ylabel('Avg. Steps  (success only)',
                       color=line_color, fontsize=9, fontweight='bold')
        ax.set_xlabel(
            r'Window Size $N$' if key == "N" else r'Threshold $\Delta d$',
            fontsize=9, labelpad=6,
        )
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)

# 底部图例
fig.legend(
    handles=[
        Patch(facecolor='#FF8C00', label='Chosen param  (N = 15  /  Δd = 1.5)'),
        Line2D([0], [0], color=line_color, marker='o',
               linewidth=2, label='Avg. Steps  (success only)'),
    ],
    loc='lower center', ncol=2, fontsize=9.5,
    bbox_to_anchor=(0.5, 0.01),
)

fig.suptitle("Hyperparameter Sensitivity Analysis:  L2 vs L5",
             fontsize=14, fontweight='bold')

# rect 为图例留底部空间，顶部不截断标题
fig.tight_layout(rect=[0, 0.07, 1, 0.96], h_pad=5.5, w_pad=3.0)

out = os.path.join(SAVE_DIR, "ablation_L2_vs_L5_clean.png")
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f"📊 图表已保存 → {out}")
plt.show()
