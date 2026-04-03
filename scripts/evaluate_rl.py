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

def evaluate_and_plot():
    print("🔍 开始加载模型与环境进行评估...")
    
    # 路径配置
    # 加载最高难度地图和最高学历模型
    map_file = os.path.join(project_root, "data", "csv_maps", "Phase1_Map_Level_5.csv")
    model_path = os.path.join(project_root, "models", "saved_weights", "ppo_curriculum_level_3.zip")
    
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型权重文件: {model_path}")
        return

    # 1. 初始化环境 (必须与训练时的结构一模一样)
    env = GridNavEnv(map_path=map_file, max_steps=200)
    eval_env = gym.wrappers.FlattenObservation(env)
    
    # 2. 加载训练好的 PPO 模型
    model = PPO.load(model_path)
    
    # 3. 运行单个测试回合，并记录轨迹
    obs, info = eval_env.reset()
    trajectory = [env.agent_pos.copy()]  # 记录起点
    total_reward = 0.0
    done = False
    
    while not done:
        # deterministic=True 表示采用最高置信度的动作，不再随机探索
        action, _states = model.predict(obs, deterministic=True)
        # 将 numpy 数组强制转换为 Python 原生整数
        action_int = int(action.item())
        obs, reward, terminated, truncated, info = eval_env.step(action)
        
        total_reward += reward
        trajectory.append(env.agent_pos.copy())
        
        done = terminated or truncated

    # 4. 打印评估结果
    print("-" * 50)
    print(f"🏁 评估完成！总耗费步数: {len(trajectory) - 1}")
    print(f"💰 累计获得奖励: {total_reward:.2f}")
    
    if terminated and reward >= 100:
        print("🎉 结果: 成功到达终点！")
    elif terminated:
        print("💥 结果: 撞墙终止。")
    else:
        print("⏳ 结果: 步数耗尽 (未能在规定步数内到达)。")
    print("-" * 50)

    # 5. 绘制并导出轨迹图 (对标你之前的 MATLAB 风格)
    print("🎨 正在生成并保存轨迹图...")
    plt.figure(figsize=(6, 6))
    
    # 绘制背景地图 (转置以符合常规的 XY 视觉习惯，0为白，1为灰/黑)
    cmap = plt.cm.colors.ListedColormap(['white', '#8A7B9D']) # 柔和的紫色障碍物
    plt.imshow(env.global_map.T, cmap=cmap, origin='lower')
    
    # 提取轨迹的 X 和 Y 坐标
    traj_x = [p[0] for p in trajectory]
    traj_y = [p[1] for p in trajectory]
    
    # 绘制轨迹连线
    plt.plot(traj_x, traj_y, color='#1f77b4', linewidth=2.5, label='RL Trajectory')
    
    # 绘制起点和终点
    plt.plot(trajectory[0][0], trajectory[0][1], 'ro', markersize=10, markeredgecolor='k', label='Start')
    plt.plot(env.goal_pos[0], env.goal_pos[1], 'rs', markersize=10, markeredgecolor='k', label='Goal')
    
    plt.axis('equal')
    plt.axis('off') # 去除坐标轴
    plt.legend(loc='lower right')
    
    # 确保保存目录存在
    fig_dir = os.path.join(project_root, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    fig_path = os.path.join(fig_dir, "RL_Level_1_Trajectory.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"✅ 轨迹图已成功导出至: {fig_path}")

if __name__ == "__main__":
    evaluate_and_plot()
