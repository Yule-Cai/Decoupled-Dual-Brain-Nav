# scripts/evaluate_moe.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import gymnasium as gym
from stable_baselines3 import PPO

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from envs.grid_nav_env import GridNavEnv
from models.moe_gating import MoEArbiter

def evaluate_moe_agent_waypoint():
    print("🧠 正在启动终极双脑系统 (卸下铠甲的完全体) ...")
    
    map_file = os.path.join(project_root, "data", "csv_maps", "Phase1_Map_Level_5.csv")
    model_path = os.path.join(project_root, "models", "saved_weights", "ppo_curriculum_level_3.zip")
    
    env = GridNavEnv(map_path=map_file, max_steps=800, add_dynamic_ob=True)
    eval_env = gym.wrappers.FlattenObservation(env)
    
    rl_model = PPO.load(model_path)
    arbiter = MoEArbiter(rl_model=rl_model, patience=10)
    
    obs, info = eval_env.reset()
    
    def get_nearest_empty(target_x, target_y):
        for radius in range(10):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = target_x + dx, target_y + dy
                    if 0 <= nx < env.map_size and 0 <= ny < env.map_size:
                        if env.global_map[nx, ny] == 0 and not (env.add_dynamic_ob and 35 < nx < 45 and 20 < ny < 60):
                            return np.array([nx, ny])
        return np.array([target_x, target_y])

    env.agent_pos = get_nearest_empty(2, 2)
    env.goal_pos = get_nearest_empty(env.map_size - 3, env.map_size - 3)
    env.previous_distance = np.linalg.norm(env.agent_pos - env.goal_pos)
    
    dict_obs = env._get_obs()
    obs = gym.spaces.flatten(env.observation_space, dict_obs)

    trajectory_data = [(env.agent_pos.copy(), "RL")] 
    done = False
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title('终极 MoE 架构实时监控 (完全体)')
    
    cmap = plt.cm.colors.ListedColormap(['white', '#8A7B9D'])
    ax.imshow(env.global_map.T, cmap=cmap, origin='lower')
    ax.plot(env.agent_pos[0], env.agent_pos[1], 'ro', markersize=12, label='Start')
    ax.plot(env.goal_pos[0], env.goal_pos[1], 'rs', markersize=12, label='True Goal')
    current_pos_plot, = ax.plot(env.agent_pos[0], env.agent_pos[1], 'go', markersize=10, zorder=5)
    
    ax.plot([], [], color='#1f77b4', linewidth=2.0, label='RL (Goal Directed)')
    ax.plot([], [], color='#ff7f0e', linewidth=3.0, label='RL (Waypoint Tracking)') 
    ax.plot([], [], 'c--', alpha=0.5, label='Intervention Link')
    waypoint_plot, = ax.plot([], [], 'y*', markersize=15, zorder=4)
    past_lines = []

    ax.axis('equal')
    ax.axis('off')
    ax.legend(loc='lower right')
    plt.show()

    while not done:
        if env.add_dynamic_ob and hasattr(env, 'dy_ob_pos'):
            if not hasattr(ax, 'dy_ob_patch'):
                h_s = env.dy_ob_size // 2
                ax.dy_ob_patch = Rectangle((env.dy_ob_pos[0]-h_s-0.5, env.dy_ob_pos[1]-h_s-0.5), env.dy_ob_size, env.dy_ob_size, color='red', fill=True, zorder=4)
                ax.add_patch(ax.dy_ob_patch)
            else:
                h_s = env.dy_ob_size // 2
                ax.dy_ob_patch.set_xy((env.dy_ob_pos[0]-h_s-0.5, env.dy_ob_pos[1]-h_s-0.5))

        action, controller = arbiter.predict(obs, env.agent_pos, env.goal_pos, env.global_map)
        
        orig_dx, orig_dy = env.action_mapping[action] 
        intended_pos = env.agent_pos + np.array([orig_dx, orig_dy])
        
        # ====== 净化护盾：只保留最基础的防撞墙和防撞车 ======
        will_hit_wall = (intended_pos[0] < 0 or intended_pos[0] >= env.map_size or 
                         intended_pos[1] < 0 or intended_pos[1] >= env.map_size or 
                         env.global_map[intended_pos[0], intended_pos[1]] == 1)
                         
        will_hit_dyn = False
        if env.add_dynamic_ob:
            tox, toy = env.dy_ob_pos
            hs = env.dy_ob_size // 2
            if tox - hs - 1 < intended_pos[0] < tox + hs + 1 and toy - hs - 1 < intended_pos[1] < toy + hs + 1:
                will_hit_dyn = True
                
        if will_hit_wall or will_hit_dyn:
            # 撞墙红叉，撞车洋红叉
            marker = 'rx' if will_hit_wall else 'mX' 
            ax.plot(intended_pos[0], intended_pos[1], marker, markersize=10, zorder=6)
            
            safe_actions = []
            for a, (ddx, ddy) in env.action_mapping.items():
                cand = env.agent_pos + np.array([ddx, ddy])
                if not (cand[0] < 0 or cand[0] >= env.map_size or cand[1] < 0 or cand[1] >= env.map_size or env.global_map[cand[0], cand[1]] == 1):
                    cand_hit_dyn = False
                    if env.add_dynamic_ob and (tox - hs - 1 < cand[0] < tox + hs + 1 and toy - hs - 1 < cand[1] < toy + hs + 1):
                        cand_hit_dyn = True
                    
                    if not cand_hit_dyn:
                        safe_actions.append(a)

            # 智能滑行：寻找与大脑原始意图最接近的安全路口
            if safe_actions:
                best_action = safe_actions[0]
                max_dot = -float('inf')
                for sa in safe_actions:
                    sdx, sdy = env.action_mapping[sa]
                    dot_product = orig_dx * sdx + orig_dy * sdy
                    if dot_product > max_dot:
                        max_dot = dot_product
                        best_action = sa
                action = best_action
                controller += " [Momentum Shield]"
            else:
                action = 0 
        # =======================================================
        
        obs, reward, terminated, truncated, info = eval_env.step(action)
        trajectory_data.append((env.agent_pos.copy(), controller))
        
        # 可视化青色介入连线
        while len(past_lines) < len(arbiter.intervention_logs):
            log = arbiter.intervention_logs[len(past_lines)]
            line, = ax.plot([log[0][0], log[1][0]], [log[0][1], log[1][1]], 'c--', alpha=0.5, zorder=2)
            past_lines.append(line)

        pos1 = trajectory_data[-2][0]
        pos2 = trajectory_data[-1][0]
        
        is_waypoint = "Waypoint" in controller
        color = '#ff7f0e' if is_waypoint else '#1f77b4'
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], color=color, linewidth=3.0 if is_waypoint else 2.0, zorder=3 if is_waypoint else 2)
        current_pos_plot.set_data([pos2[0]], [pos2[1]])
        
        if arbiter.current_waypoint is not None:
            waypoint_plot.set_data([arbiter.current_waypoint[0]], [arbiter.current_waypoint[1]])
        else:
            waypoint_plot.set_data([], [])
        
        plt.pause(0.02) 
        done = terminated or truncated

    plt.ioff()
    print("-" * 50)
    if terminated and reward >= 100:
        print("🎉 结果: 终极双脑系统成功合作到达终点！！！")
    plt.show()

if __name__ == "__main__":
    evaluate_moe_agent_waypoint()
