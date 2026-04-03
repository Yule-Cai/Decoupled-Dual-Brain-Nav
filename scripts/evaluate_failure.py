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

def evaluate_level5_failure():
    print("😈 开始加载地狱难度 (Level 5) 地图，准备获取反面教材截图...")
    
    # 【关键修改 1】换成 Level 5 的极度复杂地图 (80x80)
    map_file = os.path.join(project_root, "data", "csv_maps", "Phase1_Map_Level_5.csv")
    
    # 依然加载在 Level 1 (简单环境) 训练出的初级模型
    model_path = os.path.join(project_root, "models", "saved_weights", "ppo_level_1_nav.zip")
    
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型权重文件: {model_path}")
        return

    # 【关键修改 2】把 max_steps 调大到 800，给它充足的时间去"碰壁"和"原地打转"
    env = GridNavEnv(map_path=map_file, max_steps=800)
    eval_env = gym.wrappers.FlattenObservation(env)
    
    # 加载训练好的 PPO 模型
    model = PPO.load(model_path)
    
    # 运行测试
    obs, info = eval_env.reset()
    trajectory = [env.agent_pos.copy()]
    total_reward = 0.0
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        # 兼容动作类型
        if hasattr(action, 'item'): action_int = int(action.item())
        else: action_int = int(action)
            
        obs, reward, terminated, truncated, info = eval_env.step(action_int)
        
        total_reward += reward
        trajectory.append(env.agent_pos.copy())
        done = terminated or truncated

    # 打印评估结果
    print("-" * 50)
    print(f"🏁 评估完成！总耗费步数: {len(trajectory) - 1}")
    print(f"💰 累计获得奖励: {total_reward:.2f}")
    
    if terminated and reward >= 100:
        print("🎉 结果: 居然奇迹般地到达终点了？！(概率极低)")
    elif terminated:
        print("💥 结果: 撞墙终止。(陷入死胡同绝望自杀)")
    else:
        print("⏳ 结果: 步数耗尽 (在局部最优里疯狂打转直到超时)。")
    print("-" * 50)

    # 绘制并导出轨迹图
    print("🎨 正在生成反面教材轨迹图...")
    plt.figure(figsize=(8, 8)) # 因为是 80x80，把画布调大一点
    
    cmap = plt.cm.colors.ListedColormap(['white', '#8A7B9D'])
    plt.imshow(env.global_map.T, cmap=cmap, origin='lower')
    
    traj_x = [p[0] for p in trajectory]
    traj_y = [p[1] for p in trajectory]
    
    # 用醒目的橙色或红色表示失败轨迹
    plt.plot(traj_x, traj_y, color='#ff7f0e', linewidth=2.0, alpha=0.8, label='Failed RL Trajectory')
    
    plt.plot(trajectory[0][0], trajectory[0][1], 'ro', markersize=10, markeredgecolor='k', label='Start')
    plt.plot(env.goal_pos[0], env.goal_pos[1], 'rs', markersize=10, markeredgecolor='k', label='Goal')
    
    plt.axis('equal')
    plt.axis('off')
    plt.legend(loc='lower right')
    
    fig_dir = os.path.join(project_root, "results", "figures")
    fig_path = os.path.join(fig_dir, "RL_Level_5_Failure.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"✅ 反面教材已成功导出至: {fig_path}")

if __name__ == "__main__":
    evaluate_level5_failure()
