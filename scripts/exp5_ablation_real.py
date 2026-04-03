# scripts/exp5_ablation_real.py
# 双地图超参数消融实验 —— L3 vs L5，生成 2×2 对比图
# 用法：python scripts/exp5_ablation_real.py

import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from envs.grid_nav_env import GridNavEnv
from models.moe_gating import MoEArbiter

# ================= 配置区 =================
MAPS = {
    "L2": os.path.join(project_root, "data", "csv_maps", "Phase1_Map_Level_2.csv"),
    "L5": os.path.join(project_root, "data", "csv_maps", "Phase1_Map_Level_5.csv"),
}
MODEL_PATH = os.path.join(project_root, "models", "saved_weights", "ppo_curriculum_level_3.zip")
N_EPISODES  = 15
MAX_STEPS   = 801
SAVE_DIR    = os.path.join(project_root, "results")
os.makedirs(SAVE_DIR, exist_ok=True)

N_VALUES    = [5, 10, 15, 20, 30]
D_VALUES    = [0.5, 1.0, 1.5, 2.0, 3.0]
FIXED_DELTA = 1.5
FIXED_N     = 15
GOAL_SHIELD = 5.0
# ==========================================


def run_episodes(map_file, n_episodes, patience, delta_d, goal_shield, rl_model):
    results = {"success": 0, "steps": [], "llm_calls": []}

    for ep in range(n_episodes):
        env = GridNavEnv(map_path=map_file, max_steps=MAX_STEPS, add_dynamic_ob=False)
        eval_env = gym.wrappers.FlattenObservation(env)
        eval_env.reset()

        np.random.seed(ep * 42)
        env.agent_pos = _get_random_empty(env)
        env.goal_pos  = _get_random_empty(env)
        while np.linalg.norm(env.agent_pos - env.goal_pos) < 20:
            env.goal_pos = _get_random_empty(env)
        env.previous_distance = np.linalg.norm(env.agent_pos - env.goal_pos)

        obs     = gym.spaces.flatten(env.observation_space, env._get_obs())
        arbiter = MoEArbiter(rl_model=rl_model, patience=patience)

        def patched_predict(obs_flat, agent_pos, goal_pos, global_map,
                            _delta=delta_d, _goal=goal_shield, _arb=arbiter):
            _arb.pos_history.append(agent_pos.copy())
            if len(_arb.pos_history) > _arb.patience:
                _arb.pos_history.pop(0)
            lg = obs_flat[:49]

            if _arb.mode == "LLM_WAYPOINT":
                _arb.waypoint_steps += 1
                if np.linalg.norm(agent_pos - _arb.current_waypoint) <= 1.5 \
                        or _arb.waypoint_steps > _arb.max_waypoint_steps:
                    _arb.mode = "RL"
                    _arb.current_waypoint = None
                    _arb.pos_history.clear()
                    return _arb._get_rl_action_with_tabu(obs_flat, agent_pos, goal_pos)
                v = (_arb.current_waypoint - agent_pos).astype(np.float32)
                d = np.linalg.norm(v)
                if d > 0: v /= d
                a, _ = _arb.rl_model.predict(np.concatenate([lg, v]), deterministic=True)
                return (int(a.item()) if hasattr(a, 'item') else int(a)), "Waypoint"

            if _arb.mode == "RL":
                if np.linalg.norm(agent_pos - goal_pos) >= _goal \
                        and len(_arb.pos_history) == _arb.patience:
                    if np.linalg.norm(_arb.pos_history[-1] - _arb.pos_history[0]) < _delta:
                        _arb.mode = "LLM_WAYPOINT"
                        _arb.waypoint_steps = 0
                        _arb.trap_zones.append(agent_pos.copy())
                        try:
                            wp = _arb.llm_nav.get_waypoint(global_map, agent_pos, goal_pos, _arb.history_wps)
                        except TypeError:
                            wp = _arb.llm_nav.get_waypoint(global_map, agent_pos, goal_pos)
                        _arb.current_waypoint = wp
                        _arb.history_wps.append(wp.copy())
                        _arb.intervention_logs.append((agent_pos.copy(), wp.copy()))
                        _arb.pos_history.clear()
                        v = (wp - agent_pos).astype(np.float32)
                        d = np.linalg.norm(v)
                        if d > 0: v /= d
                        a, _ = _arb.rl_model.predict(np.concatenate([lg, v]), deterministic=True)
                        return (int(a.item()) if hasattr(a, 'item') else int(a)), "Waypoint"

            return _arb._get_rl_action_with_tabu(obs_flat, agent_pos, goal_pos)

        arbiter.predict = patched_predict

        done, step, success = False, 0, False
        while not done:
            action, _ = arbiter.predict(obs, env.agent_pos, env.goal_pos, env.global_map)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            step += 1
            done = terminated or truncated
            if terminated and reward >= 100:
                success = True

        if success:
            results["success"] += 1
            results["steps"].append(step)
        results["llm_calls"].append(len(arbiter.intervention_logs))
        print(f"    ep {ep+1:02d}  {'✅' if success else '❌'}  "
              f"steps={step}  calls={len(arbiter.intervention_logs)}")

    sr        = results["success"] / n_episodes * 100
    avg_steps = float(np.mean(results["steps"])) if results["steps"] else float('nan')
    avg_calls = float(np.mean(results["llm_calls"]))
    return {"sr": sr, "avg_steps": avg_steps, "avg_calls": avg_calls}


def _get_random_empty(env):
    while True:
        x, y = np.random.randint(0, env.map_size, 2)
        if env.global_map[x, y] == 0:
            return np.array([x, y])


def main():
    print("📦 加载 PPO 模型...")
    rl_model   = PPO.load(MODEL_PATH)
    all_results = {}

    for map_name, map_file in MAPS.items():
        print(f"\n{'='*60}\n🗺  地图: {map_name}\n{'='*60}")
        all_results[map_name] = {"N": {}, "D": {}}

        print(f"\n  🔬 消融 N (固定 Δd={FIXED_DELTA})")
        for N in N_VALUES:
            print(f"  ▶ N={N}")
            res = run_episodes(map_file, N_EPISODES, N, FIXED_DELTA, GOAL_SHIELD, rl_model)
            all_results[map_name]["N"][str(N)] = res
            print(f"  → SR={res['sr']:.0f}%  steps={res['avg_steps']:.1f}  calls={res['avg_calls']:.2f}")

        print(f"\n  🔬 消融 Δd (固定 N={FIXED_N})")
        for D in D_VALUES:
            print(f"  ▶ Δd={D}")
            res = run_episodes(map_file, N_EPISODES, FIXED_N, D, GOAL_SHIELD, rl_model)
            all_results[map_name]["D"][str(D)] = res
            print(f"  → SR={res['sr']:.0f}%  steps={res['avg_steps']:.1f}  calls={res['avg_calls']:.2f}")

    def convert(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    save_path = os.path.join(SAVE_DIR, "ablation_dual_map.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, default=convert, indent=2, ensure_ascii=False)
    print(f"\n💾 原始数据 → {save_path}")

    plot_dual(all_results)


def plot_dual(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor('#fafafa')
    fig.suptitle("Hyperparameter Sensitivity: L2 vs L5  (steps ↓ = better)",
                 fontsize=13, fontweight='bold', y=1.01)

    map_labels = {"L2": "Level 2 (Trap-Heavy)", "L5": "Level 5 (Hard)"}
    col_titles = [r"Sensitivity to $N$  ($\Delta d=1.5$ fixed)",
                  r"Sensitivity to $\Delta d$  ($N=15$ fixed)"]
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
            steps = [data[k]['avg_steps'] if not np.isnan(data[k]['avg_steps']) else 0 for k in ks]
            srs   = [data[k]['sr'] for k in ks]

            hi     = HIGHLIGHT[key]
            colors = [('#FF8C00' if k == hi else bar_colors[col]) for k in ks]
            bars   = ax.bar(ks, calls, color=colors, alpha=0.82, width=0.5, zorder=2)

            ax.set_ylabel('Avg. LLM Calls / Ep', color=bar_colors[col],
                          fontsize=9, fontweight='bold')
            ax.tick_params(axis='y', labelcolor=bar_colors[col], labelsize=8)

            ax2.plot(ks, steps, color=line_color, marker='o',
                     linewidth=2.5, markersize=7, zorder=3)
            ax2.set_ylabel('Avg. Steps', color=line_color, fontsize=9, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor=line_color, labelsize=8)
            ax2.spines['top'].set_visible(False)

            # SR% 标注在柱顶
            for bar, sr in zip(bars, srs):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(calls) * 0.04,
                        f"SR\n{sr:.0f}%", ha='center', va='bottom',
                        fontsize=7, color='#333')

            # 标注推荐参数
            if hi in ks:
                i = ks.index(hi)
                ax2.annotate('★ Chosen',
                             xy=(ks[i], steps[i]),
                             xytext=(ks[i], steps[i] + max(steps) * 0.15),
                             arrowprops=dict(facecolor='#FF8C00', shrink=0.05,
                                             width=1.5, headwidth=6),
                             ha='center', fontsize=8, color='#FF8C00', fontweight='bold')

            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight='bold', pad=10)
            if col == 0:
                ax.set_ylabel(f"{map_labels[map_name]}\n\nAvg. LLM Calls / Ep",
                              color=bar_colors[col], fontsize=9, fontweight='bold')

            ax.set_xlabel(r'Window Size $N$' if key == "N" else r'Threshold $\Delta d$',
                          fontsize=9)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Patch(facecolor='#FF8C00', label='Chosen param (N=15 / Δd=1.5)'),
        Line2D([0], [0], color=line_color, marker='o', linewidth=2, label='Avg. Steps'),
    ], loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(pad=2.5)
    out = os.path.join(SAVE_DIR, "ablation_L3_vs_L5.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"📊 对比图 → {out}")
    plt.show()


if __name__ == "__main__":
    main()
