# scripts/train_curriculum.py
import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# 将项目根目录加入系统路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from envs.grid_nav_env import GridNavEnv

def train_curriculum_agent():
    print("🎓 开始启动 LFM-RL 课程学习 (600万步史诗级训练) 流水线...")
    
    # 课程设置：三个难度递增的地图
    levels = ["Level_1", "Level_2", "Level_3"]
    # 每个阶段训练 200 万步，总计 600 万步！
    steps_per_level = 2000000  
    
    # 目录准备
    model_save_dir = os.path.join(project_root, "models", "saved_weights")
    checkpoint_dir = os.path.join(project_root, "models", "checkpoints")
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    current_model_path = None # 用于在不同学期之间传递大脑权重
    
    for level in levels:
        print(f"\n{'='*50}")
        print(f"🏫 正在进入新学期: {level}")
        print(f"{'='*50}")
        
        map_file = os.path.join(project_root, "data", "csv_maps", f"Phase1_Map_{level}.csv")
        if not os.path.exists(map_file):
            print(f"❌ 找不到地图文件: {map_file}，跳过此阶段。请确保生成了该地图。")
            continue
            
        # 1. 构建环境 (最大步数放宽到 400，给它充足的探索时间)
        def make_env():
            env = GridNavEnv(map_path=map_file, max_steps=400)
            return gym.wrappers.FlattenObservation(env)
            
        vec_env = DummyVecEnv([make_env])
        
        # 2. 设置断点保存回调函数 (每跑 50 万步自动存一次档)
        checkpoint_callback = CheckpointCallback(
            save_freq=500000, 
            save_path=os.path.join(checkpoint_dir, level),
            name_prefix=f"ppo_curr_{level}"
        )
        
        # 3. 模型加载与继承逻辑
        if current_model_path is None:
            # 第一关 (小学)：从零开始初始化大脑
            print("🧠 初始化初始策略网络 (MlpPolicy)...")
            model = PPO(
                "MlpPolicy", 
                vec_env, 
                verbose=1,
                n_steps=2048,
                batch_size=128,
                learning_rate=3e-4,
                device="cpu", # 在纯 CPU 上运行
                tensorboard_log=os.path.join(project_root, "results", "logs")
            )
        else:
            # 后续关卡 (中学/大学)：加载上一关的毕业权重，继续深造
            print(f"🔄 继承上一阶段记忆，加载权重: {current_model_path}.zip")
            model = PPO.load(current_model_path, env=vec_env, device="cpu")
            
        # 4. 开始当前关卡的疯狂训练
        print(f"⚔️ 开始在 {level} 环境下激战 {steps_per_level} 步...")
        model.learn(
            total_timesteps=steps_per_level, 
            tb_log_name=f"Curriculum_PPO_{level}", 
            callback=checkpoint_callback, # 注入断点保存机制
            reset_num_timesteps=False
        )
        
        # 5. 保存当前关卡毕业的最终模型
        save_path = os.path.join(model_save_dir, f"ppo_curriculum_{level.lower()}")
        model.save(save_path)
        current_model_path = save_path
        
        print(f"✅ {level} 阶段毕业！模型已保存至: {save_path}.zip")
        vec_env.close()

    print("\n🎉 恭喜！600万步的史诗级课程学习圆满结束！你的 RL 智能体现在拥有极强的避障本能了！")

if __name__ == "__main__":
    train_curriculum_agent()
