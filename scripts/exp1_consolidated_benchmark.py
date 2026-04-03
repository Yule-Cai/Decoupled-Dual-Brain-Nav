# scripts/exp1_consolidated_benchmark.py
import os
import sys
import time
import numpy as np
import pandas as pd
import heapq
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
import gymnasium as gym
from stable_baselines3 import PPO

# 环境路径设置
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入您的自定义模块
from envs.grid_nav_env import GridNavEnv
from models.moe_gating import MoEArbiter

# ==========================================
# 核心算法组件 (从您的原脚本整合)
# ==========================================

class GraphPlanners:
    def __init__(self, internal_map):
        self.grid = internal_map
        self.max_x, self.max_y = internal_map.shape
        self.motions = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)
        ]

    def heuristic(self, p1, p2):
        dx, dy = abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
        return 1.0 * (dx + dy) + (1.414 - 2 * 1.0) * min(dx, dy)

    def search(self, start, goal, w_g=1.0, w_h=1.0):
        start_node, goal_node = tuple(start), tuple(goal)
        open_set = []
        heapq.heappush(open_set, (0.0, start_node))
        came_from = {}
        g_score = {start_node: 0.0}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == goal_node:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_node)
                return np.array(path[::-1]), "Success"
                
            for dx, dy, cost in self.motions:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < self.max_x and 0 <= neighbor[1] < self.max_y) or self.grid[neighbor[0], neighbor[1]] == 1:
                    continue
                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = w_g * tentative_g + w_h * self.heuristic(neighbor, goal_node)
                    heapq.heappush(open_set, (f, neighbor))
        return np.array([start]), "Trapped"

def run_blind_search_agent(algo_name, start, goal, global_map, view_radius=3, max_steps=1500):
    grid_shape = global_map.shape
    internal_map = np.zeros_like(global_map) 
    pos = start.copy()
    path = [pos.copy()]
    current_plan = []

    def sense_and_update():
        x, y = pos
        changed = False
        for dx in range(-view_radius, view_radius + 1):
            for dy in range(-view_radius, view_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_shape[0] and 0 <= ny < grid_shape[1]:
                    if internal_map[nx, ny] != global_map[nx, ny]:
                        internal_map[nx, ny] = global_map[nx, ny]
                        changed = True
        return changed

    for step in range(max_steps):
        if np.array_equal(pos, goal):
            return np.array(path), "Success"
        sense_and_update()
        need_replan = False
        if not current_plan:
            need_replan = True
        else:
            for p in current_plan:
                if internal_map[p[0], p[1]] == 1:
                    need_replan = True
                    break
        if need_replan:
            gp = GraphPlanners(internal_map)
            if algo_name == "Dijkstra": new_plan, status = gp.search(pos, goal, w_g=1.0, w_h=0.0)
            elif algo_name == "A* Search": new_plan, status = gp.search(pos, goal, w_g=1.0, w_h=1.0)
            elif algo_name == "GBFS": new_plan, status = gp.search(pos, goal, w_g=0.0, w_h=1.0)
            else: new_plan, status = gp.search(pos, goal, w_g=1.0, w_h=1.0) 
            if status != "Success" or len(new_plan) <= 1:
                return np.array(path), "Local Minima" 
            current_plan = [tuple(x) for x in new_plan[1:]]
        if not current_plan: return np.array(path), "Local Minima"
        next_node = current_plan.pop(0)
        pos = np.array(next_node)
        path.append(pos.copy())
    return np.array(path), "Timeout"

def run_apf(start, goal, grid, max_iter=1500):
    pos = start.astype(float)
    path = [start.copy()]
    for _ in range(max_iter):
        if np.linalg.norm(pos - goal) < 1.5: return np.array(path), "Success"
        vec_goal = goal - pos
        dist = np.linalg.norm(vec_goal)
        f_att = (vec_goal / dist) * 2.0 if dist > 0 else np.zeros(2)
        f_rep = np.zeros(2)
        x, y = int(round(pos[0])), int(round(pos[1]))
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] == 1:
                    obs_pos = np.array([nx, ny])
                    dist_obs = np.linalg.norm(pos - obs_pos)
                    if dist_obs < 3.0 and dist_obs > 0:
                        f_rep += ((pos - obs_pos) / (dist_obs**3)) * 5.0
        force = f_att + f_rep
        f_norm = np.linalg.norm(force)
        if f_norm > 0: force = force / f_norm * 1.0 
        next_pos = pos + force
        if grid[int(round(next_pos[0])), int(round(next_pos[1]))] == 1:
            return np.array(path), "Trapped"
        pos = next_pos
        path.append(np.round(pos).astype(int))
    return np.array(path), "Timeout"

def run_rl_agent(start, goal, map_file, model_path, use_moe=False):
    env = GridNavEnv(map_path=map_file, max_steps=800, add_dynamic_ob=False)
    eval_env = gym.wrappers.FlattenObservation(env)
    if not os.path.exists(model_path): return np.array([start]), "Model Error"
    model = PPO.load(model_path)
    obs, _ = eval_env.reset()
    env.agent_pos, env.goal_pos = start.copy(), goal.copy()
    env.previous_distance = np.linalg.norm(env.agent_pos - env.goal_pos)
    obs = gym.spaces.flatten(env.observation_space, env._get_obs())
    arbiter = MoEArbiter(rl_model=model, patience=12) if use_moe else None
    path = [start.copy()]
    done = False
    while not done:
        if use_moe:
            action, controller = arbiter.predict(obs, env.agent_pos, env.goal_pos, env.global_map)
            orig_dx, orig_dy = env.action_mapping[action]
            intended_pos = env.agent_pos + np.array([orig_dx, orig_dy])
            if (intended_pos[0] < 0 or intended_pos[0] >= env.map_size or 
                intended_pos[1] < 0 or intended_pos[1] >= env.map_size or 
                env.global_map[intended_pos[0], intended_pos[1]] == 1):
                safe_actions = []
                for a, (dx, dy) in env.action_mapping.items():
                    cand = env.agent_pos + np.array([dx, dy])
                    if not (cand[0] < 0 or cand[0] >= env.map_size or cand[1] < 0 or cand[1] >= env.map_size or env.global_map[cand[0], cand[1]] == 1):
                        safe_actions.append(a)
                if safe_actions:
                    best_action = safe_actions[0]; max_dot = -float('inf')
                    for sa in safe_actions:
                        sdx, sdy = env.action_mapping[sa]; dot_product = orig_dx * sdx + orig_dy * sdy
                        if dot_product > max_dot: max_dot = dot_product; best_action = sa
                    action = best_action
                else: action = 0
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action.item()) if hasattr(action, 'item') else int(action)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        path.append(env.agent_pos.copy())
        if info.get('hit_wall') or info.get('hit_dy_ob'): return np.array(path), "Local Minima"
        done = terminated or truncated
    return np.array(path), "Success" if terminated else "Trapped"


# ==========================================
# 可视化与表格生成模块
# ==========================================

# --- 全局样式配置 ---
ALGO_NAMES = ["Local Dijkstra", "Local A*", "Local GBFS", "APF (Reactive)", "Pure PPO", "Ours (RL+LLM)"]

PALETTE = {
    "Local Dijkstra":  "#4C72B0",
    "Local A*":        "#55A868",
    "Local GBFS":      "#C44E52",
    "APF (Reactive)":  "#8172B2",
    "Pure PPO":        "#CCB974",
    "Ours (RL+LLM)":   "#E84393",  # 高亮我方方法
}

MARKERS = {
    "Local Dijkstra":  "o",
    "Local A*":        "s",
    "Local GBFS":      "^",
    "APF (Reactive)":  "D",
    "Pure PPO":        "v",
    "Ours (RL+LLM)":   "*",
}

LINESTYLES = {
    "Local Dijkstra":  (0, (5, 2)),
    "Local A*":        (0, (3, 1, 1, 1)),
    "Local GBFS":      (0, (1, 1)),
    "APF (Reactive)":  (0, (5, 2, 1, 2)),
    "Pure PPO":        "--",
    "Ours (RL+LLM)":   "-",
}

STATUS_COLORS = {
    "Success":      "#2ecc71",
    "Local Minima": "#e67e22",
    "Timeout":      "#e74c3c",
    "Trapped":      "#9b59b6",
    "Model Error":  "#95a5a6",
}


def setup_style():
    """统一设置 matplotlib 全局样式"""
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   9,
        "legend.framealpha": 0.85,
        "figure.dpi":        150,
    })


def plot_success_rate(ax, summary):
    """折线图：各算法在不同难度地图上的成功率"""
    levels = sorted(summary["Level"].unique())
    for algo in ALGO_NAMES:
        data = summary[summary["Algorithm"] == algo].sort_values("Level")
        vals = data["SuccessRate"].values * 100
        ax.plot(data["Level"], vals,
                color=PALETTE[algo],
                marker=MARKERS[algo],
                linestyle=LINESTYLES[algo],
                linewidth=2.2 if algo == "Ours (RL+LLM)" else 1.5,
                markersize=8 if algo == "Ours (RL+LLM)" else 6,
                label=algo,
                zorder=5 if algo == "Ours (RL+LLM)" else 3)

    ax.set_title("① Success Rate across Map Levels")
    ax.set_xlabel("Map Level (Difficulty)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_xticks(levels)
    ax.set_xticklabels([f"L{l}" for l in levels])
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left", ncol=2)


def plot_avg_steps(ax, summary):
    """柱状图：成功 trial 的平均步数（效率对比）"""
    levels = sorted(summary["Level"].unique())
    n_algos = len(ALGO_NAMES)
    width = 0.12
    x = np.arange(len(levels))

    for i, algo in enumerate(ALGO_NAMES):
        data = summary[summary["Algorithm"] == algo].sort_values("Level")
        vals = data["AvgSteps"].fillna(0).values
        offset = (i - n_algos / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals,
                      width=width,
                      color=PALETTE[algo],
                      label=algo,
                      alpha=0.88,
                      edgecolor="white",
                      linewidth=0.5,
                      zorder=3)
        # 在 "Ours" 柱子顶端加数值标注
        if algo == "Ours (RL+LLM)":
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                            f"{int(v)}", ha="center", va="bottom",
                            fontsize=7.5, color=PALETTE[algo], fontweight="bold")

    ax.set_title("② Avg. Steps (Successful Trials Only)")
    ax.set_xlabel("Map Level (Difficulty)")
    ax.set_ylabel("Average Steps")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in levels])
    ax.legend(loc="upper left", ncol=2)


def plot_status_breakdown(ax, df):
    """堆叠柱状图：每个算法的 status 分布（汇总所有地图）"""
    status_order = ["Success", "Local Minima", "Trapped", "Timeout", "Model Error"]
    counts = (df.groupby(["Algorithm", "Status"])
               .size()
               .unstack(fill_value=0)
               .reindex(columns=[s for s in status_order if s in df["Status"].unique()], fill_value=0))
    counts = counts.reindex(ALGO_NAMES).fillna(0)
    totals = counts.sum(axis=1)
    pcts = counts.div(totals, axis=0) * 100

    bottom = np.zeros(len(ALGO_NAMES))
    for status in pcts.columns:
        vals = pcts[status].values
        bars = ax.barh(ALGO_NAMES, vals, left=bottom,
                       color=STATUS_COLORS.get(status, "#bdc3c7"),
                       label=status, edgecolor="white", linewidth=0.5)
        # 只在比例 > 8% 时标文字
        for bar, v in zip(bars, vals):
            if v > 8:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{v:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottom += vals

    ax.set_title("③ Outcome Distribution (All Levels)")
    ax.set_xlabel("Proportion (%)")
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", fontsize=8)
    # 高亮 Ours 行
    our_idx = ALGO_NAMES.index("Ours (RL+LLM)")
    ax.get_yticklabels()[our_idx].set_color(PALETTE["Ours (RL+LLM)"])
    ax.get_yticklabels()[our_idx].set_fontweight("bold")


def plot_summary_table(ax, summary):
    """渲染一张美观的汇总数据表格"""
    ax.axis("off")

    # 构建透视表
    sr_pivot = (summary.pivot(index="Algorithm", columns="Level", values="SuccessRate")
                .reindex(ALGO_NAMES) * 100)
    steps_pivot = (summary.pivot(index="Algorithm", columns="Level", values="AvgSteps")
                   .reindex(ALGO_NAMES))

    levels = sorted(summary["Level"].unique())

    # 表头
    col_labels = (
        ["Algorithm"] +
        [f"SR-L{l}(%)" for l in levels] +
        ["Avg SR(%)"] +
        [f"Steps-L{l}" for l in levels] +
        ["Avg Steps"]
    )

    # 填充数据行
    rows = []
    for algo in ALGO_NAMES:
        sr_vals   = [f"{sr_pivot.loc[algo, l]:.1f}" if l in sr_pivot.columns else "—"
                     for l in levels]
        avg_sr    = f"{sr_pivot.loc[algo].mean():.1f}"
        step_vals = [f"{steps_pivot.loc[algo, l]:.0f}" if (l in steps_pivot.columns and not np.isnan(steps_pivot.loc[algo, l])) else "—"
                     for l in levels]
        avg_steps = steps_pivot.loc[algo].mean()
        avg_steps_str = f"{avg_steps:.0f}" if not np.isnan(avg_steps) else "—"
        rows.append([algo] + sr_vals + [avg_sr] + step_vals + [avg_steps_str])

    # 绘制表格
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.55)

    # 美化表格
    n_cols = len(col_labels)
    n_rows = len(rows)
    header_color   = "#2c3e50"
    ours_color     = "#fde8f3"
    alt_row_color  = "#f4f6f8"
    white          = "#ffffff"

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d5d8dc")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
        elif rows[r - 1][0] == "Ours (RL+LLM)":
            cell.set_facecolor(ours_color)
            if c == 0:
                cell.set_text_props(color=PALETTE["Ours (RL+LLM)"], fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor(alt_row_color)
        else:
            cell.set_facecolor(white)

    ax.set_title("④ Full Benchmark Summary Table", fontsize=13, fontweight="bold", pad=12)


def generate_visualizations(df, summary, save_dir):
    """主可视化函数：生成完整的 2×2 图表面板 + 独立表格图"""
    setup_style()

    # ── 图 1：2×2 综合面板 ──────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#fafafa")
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97, top=0.93, bottom=0.06)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_success_rate(ax1, summary)
    plot_avg_steps(ax2, summary)
    plot_status_breakdown(ax3, df)
    plot_summary_table(ax4, summary)

    fig.suptitle("Navigation Algorithm Benchmark — Comprehensive Results",
                 fontsize=16, fontweight="bold", color="#2c3e50", y=0.97)

    panel_path = os.path.join(save_dir, "benchmark_panel.png")
    fig.savefig(panel_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  📊 综合面板已保存：{panel_path}")

    # ── 图 2：独立成功率折线图（高分辨率，适合论文）──────────────────
    fig2, ax = plt.subplots(figsize=(9, 5))
    fig2.patch.set_facecolor("white")
    plot_success_rate(ax, summary)
    ax.set_title("Success Rate across Map Levels", fontsize=14, fontweight="bold")
    sr_path = os.path.join(save_dir, "success_rate_comparison.png")
    fig2.savefig(sr_path, bbox_inches="tight", dpi=200)
    plt.close(fig2)
    print(f"  📈 成功率折线图已保存：{sr_path}")

    # ── 图 3：独立表格图 ─────────────────────────────────────────────────
    levels = sorted(summary["Level"].unique())
    n_cols = 1 + len(levels) + 1 + len(levels) + 1
    fig3, ax = plt.subplots(figsize=(max(14, n_cols * 1.4), 4.5))
    fig3.patch.set_facecolor("white")
    plot_summary_table(ax, summary)
    tbl_path = os.path.join(save_dir, "benchmark_table.png")
    fig3.savefig(tbl_path, bbox_inches="tight", dpi=200)
    plt.close(fig3)
    print(f"  📋 独立表格图已保存：{tbl_path}")


def export_latex_table(summary, save_dir):
    """额外导出 LaTeX 格式表格（方便写论文直接 paste）"""
    levels = sorted(summary["Level"].unique())
    sr_pivot = (summary.pivot(index="Algorithm", columns="Level", values="SuccessRate")
                .reindex(ALGO_NAMES) * 100)
    steps_pivot = (summary.pivot(index="Algorithm", columns="Level", values="AvgSteps")
                   .reindex(ALGO_NAMES))

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Navigation Benchmark Results}")
    lines.append(r"\label{tab:benchmark}")

    # 列格式
    col_fmt = "l" + "c" * len(levels) + "c" + "c" * len(levels) + "c"
    lines.append(r"\begin{tabular}{" + col_fmt + "}")
    lines.append(r"\toprule")

    # 多级表头
    sr_span  = f"\\multicolumn{{{len(levels)+1}}}{{c}}{{Success Rate (\\%)}}"
    st_span  = f"\\multicolumn{{{len(levels)+1}}}{{c}}{{Avg. Steps}}"
    lines.append(f"\\multirow{{2}}{{*}}{{Algorithm}} & {sr_span} & {st_span} \\\\")
    lvl_hdr  = " & ".join([f"L{l}" for l in levels] + ["Avg"] + [f"L{l}" for l in levels] + ["Avg"])
    lines.append(f"\\cmidrule(lr){{2-{1+len(levels)+1}}} \\cmidrule(lr){{{2+len(levels)+1}-{1+2*(len(levels)+1)}}}")
    lines.append(f" & {lvl_hdr} \\\\")
    lines.append(r"\midrule")

    for algo in ALGO_NAMES:
        sr_vals = [f"{sr_pivot.loc[algo, l]:.1f}" if l in sr_pivot.columns else "—" for l in levels]
        avg_sr  = f"{sr_pivot.loc[algo].mean():.1f}"
        st_vals = [f"{steps_pivot.loc[algo, l]:.0f}"
                   if (l in steps_pivot.columns and not np.isnan(steps_pivot.loc[algo, l])) else "—"
                   for l in levels]
        avg_st  = steps_pivot.loc[algo].mean()
        avg_st_str = f"{avg_st:.0f}" if not np.isnan(avg_st) else "—"

        row_vals = sr_vals + [avg_sr] + st_vals + [avg_st_str]
        # 粗体高亮 Ours
        algo_cell = f"\\textbf{{{algo}}}" if algo == "Ours (RL+LLM)" else algo
        lines.append(f"{algo_cell} & " + " & ".join(row_vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_path = os.path.join(save_dir, "benchmark_table.tex")
    with open(latex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  📄 LaTeX 表格已导出：{latex_path}")


# ==========================================
# 自动化实验引擎
# ==========================================

def run_experiment():
    num_maps   = 5
    num_trials = 20
    all_data   = []

    print(f"🚀 开始全量测试：{num_maps} 张地图 × {num_trials} 次重复...\n")

    for level in range(1, num_maps + 1):
        print(f"🗺️  正在测试地图 Level {level}...")
        map_file   = os.path.join(project_root, "data", "csv_maps", f"Phase1_Map_Level_{level}.csv")
        model_path = os.path.join(project_root, "models", "saved_weights", "ppo_curriculum_level_3.zip")

        if not os.path.exists(map_file):
            print(f"   ⚠️  地图文件不存在，跳过：{map_file}")
            continue

        global_map = pd.read_csv(map_file, header=None).values
        map_size   = global_map.shape[0]

        def get_nearest_empty(target_x, target_y):
            for r in range(10):
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        nx, ny = target_x + dx, target_y + dy
                        if 0 <= nx < map_size and 0 <= ny < map_size and global_map[nx, ny] == 0:
                            return np.array([nx, ny])
            return np.array([target_x, target_y])

        start_pos = get_nearest_empty(2, 2)
        goal_pos  = get_nearest_empty(map_size - 3, map_size - 3)

        algorithms = {
            "Local Dijkstra":  lambda: run_blind_search_agent("Dijkstra",   start_pos, goal_pos, global_map),
            "Local A*":        lambda: run_blind_search_agent("A* Search",  start_pos, goal_pos, global_map),
            "Local GBFS":      lambda: run_blind_search_agent("GBFS",       start_pos, goal_pos, global_map),
            "APF (Reactive)":  lambda: run_apf(start_pos, goal_pos, global_map),
            "Pure PPO":        lambda: run_rl_agent(start_pos, goal_pos, map_file, model_path, use_moe=False),
            "Ours (RL+LLM)":   lambda: run_rl_agent(start_pos, goal_pos, map_file, model_path, use_moe=True),
        }

        for name in ALGO_NAMES:
            print(f"   🏃 {name:18s}: ", end="", flush=True)
            success_count = 0
            for t in range(num_trials):
                path, status  = algorithms[name]()
                is_success    = 1 if status == "Success" else 0
                success_count += is_success
                all_data.append({
                    "Level": level, "Algorithm": name, "Trial": t + 1,
                    "Status": status, "Steps": len(path), "Success": is_success,
                })
                print("✓" if is_success else "✗", end="", flush=True)
            print(f"  [{success_count}/{num_trials}]  {success_count/num_trials:.0%}")

    # ── 保存原始数据 ──────────────────────────────────────────────────────
    df = pd.DataFrame(all_data)
    save_dir = os.path.join(project_root, "results")
    os.makedirs(save_dir, exist_ok=True)

    raw_path = os.path.join(save_dir, "benchmark_raw_results.csv")
    df.to_csv(raw_path, index=False)
    print(f"\n💾 原始数据已保存：{raw_path}")

    # ── 汇总统计 ──────────────────────────────────────────────────────────
    summary = (
        df.groupby(["Level", "Algorithm"])
        .apply(lambda g: pd.Series({
            "SuccessRate": g["Success"].mean(),
            "AvgSteps":    g.loc[g["Success"] == 1, "Steps"].mean(),
            "StdSteps":    g.loc[g["Success"] == 1, "Steps"].std(),
            "TrialCount":  len(g),
        }))
        .reset_index()
    )
    summary_path = os.path.join(save_dir, "benchmark_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"💾 汇总数据已保存：{summary_path}")

    # ── 可视化 & 表格生成 ─────────────────────────────────────────────────
    print("\n🎨 正在生成可视化图表与表格...")
    generate_visualizations(df, summary, save_dir)
    export_latex_table(summary, save_dir)

    print(f"\n✅ 全部完成！所有结果已保存至 {save_dir}/")
    print("   生成文件：")
    print("     benchmark_raw_results.csv   — 完整原始数据")
    print("     benchmark_summary.csv       — 汇总统计数据")
    print("     benchmark_panel.png         — 2×2 综合图表面板")
    print("     success_rate_comparison.png — 独立成功率折线图（高清）")
    print("     benchmark_table.png         — 独立汇总表格图")
    print("     benchmark_table.tex         — LaTeX 表格（可直接粘贴到论文）")


if __name__ == "__main__":
    run_experiment()
