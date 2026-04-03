# scripts/exp3_ppo_entropy_trap.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import torch
from stable_baselines3 import PPO

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from envs.grid_nav_env import GridNavEnv

def run_entropy_analysis():
    # 我们故意选一个最容易卡死的难度 (Map 4)
    map_path = os.path.join(project_root, "data", "csv_maps", "Phase1_Map_Level_4.csv")
    env = GridNavEnv(map_path=map_path, max_steps=200) # 限制200步足够看清卡死状态
    eval_env = gym.wrappers.FlattenObservation(env)
    
    model_path = os.path.join(project_root, "models", "saved_weights", "ppo_curriculum_level_3.zip")
    model = PPO.load(model_path)
    
    obs, _ = eval_env.reset()
    entropies = []
    distances = []
    
    done = False
    step = 0
    
    print("🏃 开始抓取 PPO 撞墙时的策略熵数据...")
    while not done:
        # 1. 提取底层策略分布，计算当前动作的策略熵 (Policy Entropy)
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        distribution = model.policy.get_distribution(obs_tensor)
        entropy = distribution.entropy().mean().item()
        
        # 2. 获取正常的动作
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = eval_env.step(action)
        
        dist_to_goal = np.linalg.norm(env.agent_pos - env.goal_pos)
        
        entropies.append(entropy)
        distances.append(dist_to_goal)
        
        step += 1
        done = term or trunc

    # 绘图：策略熵 vs 距离终点的距离
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = 'tab:red'
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Distance to Goal (Grid Units)', color=color)
    ax1.plot(distances, color=color, linewidth=2, label="Distance to Goal")
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Policy Entropy (Confidence)', color=color)  
    ax2.plot(entropies, color=color, linestyle='--', linewidth=2, label="Policy Entropy")
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("PPO Agent Trapped in Local Minima: The 'Overconfidence' Trap", fontweight='bold')
    fig.tight_layout()
    
    save_path = os.path.join(project_root, "results", "ppo_entropy_trap.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"✅ 完美！熵值崩塌证据图已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_entropy_analysis()
