# scripts/evaluate_expert_rl.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO

# 将项目根目录加入系统路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from envs.grid_nav_env import GridNavEnv

def evaluate_expert():
    print("🧠 正在唤醒 600 万步特种兵 (纯 RL 极限测试) ...")
    
    # 直接挑战最复杂的地狱难度 Level 5
    map_file = os.path.join(project_root, "data", "csv_maps", "Phase1_Map_Level_5.csv")
    model_path = os.path.join(project_root, "models", "saved_weights", "ppo_curriculum_level_3.zip")
    
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型权重: {model_path}")
        return

    # 初始化环境
    env = GridNavEnv(map_path=map_file, max_steps=800)
    eval_env = gym.wrappers.FlattenObservation(env)
    
    model = PPO.load(model_path)
    
    # 1. 先进行环境默认的重置
    obs, info = eval_env.reset()
    
    # ==================== 新增：强制对角线极限挑战 ====================
    # 写一个内部小函数：在指定坐标附近找一个不是墙壁的空地
    def get_nearest_empty(target_x, target_y):
        for radius in range(10):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = target_x + dx, target_y + dy
                    if 0 <= nx < env.map_size and 0 <= ny < env.map_size:
                        if env.global_map[nx, ny] == 0:
                            return np.array([nx, ny])
        return np.array([target_x, target_y])

    # 强行把机器人按在左下角，把目标扔到右上角！
    env.agent_pos = get_nearest_empty(2, 2)
    env.goal_pos = get_nearest_empty(env.map_size - 3, env.map_size - 3)
    
    # 重新计算距离
    env.previous_distance = np.linalg.norm(env.agent_pos - env.goal_pos)
    
    # 【极度关键】：获取字典格式的 obs，并用 Gymnasium 工具手动展平为 51 维向量
    dict_obs = env._get_obs()
    obs = gym.spaces.flatten(env.observation_space, dict_obs)
    # =================================================================
    
    trajectory = [env.agent_pos.copy()]
    done = False
    
    print("-" * 50)
    print("🚀 开始特种兵单人极限挑战 (对角线长征)！请观察弹出的实时画面...")
    print("-" * 50)

    # 实时可视化设置
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title('600万步 RL 纯享版极限测试')
    
    cmap = plt.cm.colors.ListedColormap(['white', '#8A7B9D'])
    ax.imshow(env.global_map.T, cmap=cmap, origin='lower')
    ax.plot(trajectory[0][0], trajectory[0][1], 'ro', markersize=12, markeredgecolor='k', label='Start')
    ax.plot(env.goal_pos[0], env.goal_pos[1], 'rs', markersize=12, markeredgecolor='k', label='Goal')
    
    current_pos_plot, = ax.plot(env.agent_pos[0], env.agent_pos[1], 'go', markersize=10, zorder=5)
    line_plot, = ax.plot([], [], color='#1f77b4', linewidth=2.5, zorder=3, label='Expert RL')
    
    ax.axis('equal')
    ax.axis('off')
    ax.legend(loc='lower right')
    plt.show()

    # 开始跑酷
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        if hasattr(action, 'item'): action = int(action.item())
        else: action = int(action)
            
        obs, reward, terminated, truncated, info = eval_env.step(action)
        trajectory.append(env.agent_pos.copy())
        
        # 更新轨迹线和当前绿点
        traj_x = [p[0] for p in trajectory]
        traj_y = [p[1] for p in trajectory]
        line_plot.set_data(traj_x, traj_y)
        current_pos_plot.set_data([env.agent_pos[0]], [env.agent_pos[1]])
        
        plt.pause(0.01) # 极速播放
        done = terminated or truncated

    plt.ioff()
    print("-" * 50)
    if terminated and reward >= 100:
        print("🎉 结果: 纯 RL 凭借肌肉记忆单刷 Level 5 成功！！！")
    else:
        print("💥 结果: 陷入了无法仅凭直觉逃脱的超级迷宫。")
    print("-" * 50)
    plt.show()

if __name__ == "__main__":
    evaluate_expert()
