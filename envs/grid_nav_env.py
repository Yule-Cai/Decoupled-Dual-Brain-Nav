# envs/grid_nav_env.py
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

class GridNavEnv(gym.Env):
    """
    终极动态安全环境 (Dynamic Obstacle + Safety Check)
    状态特征：0=空地, 1=静态墙壁, 0.5=动态障碍物
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, map_path=None, max_steps=400, add_dynamic_ob=True):
        super().__init__()
        
        if map_path and os.path.exists(map_path):
            self.global_map = pd.read_csv(map_path, header=None).values
        else:
            self.global_map = np.zeros((20, 20))
            
        self.map_size = self.global_map.shape[0]
        self.max_steps = max_steps
        self.add_dynamic_ob = add_dynamic_ob
        
        self.action_space = spaces.Discrete(8)
        self.action_mapping = {
            0: np.array([-1, 0]), 1: np.array([1, 0]), 
            2: np.array([0, -1]), 3: np.array([0, 1]),
            4: np.array([-1, -1]), 5: np.array([-1, 1]), 
            6: np.array([1, -1]), 7: np.array([1, 1])
        }
        
        self.observation_space = spaces.Dict({
            "local_grid": spaces.Box(low=0.0, high=1.0, shape=(7, 7), dtype=np.float32),
            "target_vector": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        })
        
        self.agent_pos = None
        self.goal_pos = None
        self.current_step = 0
        self.previous_distance = 0.0
        
        # ====== 修复：将巡逻车放在地图绝对正中央 (40, 40) ======
        if self.add_dynamic_ob:
            self.dy_ob_size = 3 
            self.dy_ob_pos = np.array([40, 40]) 
            self.dy_ob_speed = 1 
            self.dy_ob_direction = 1 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.agent_pos = self._get_random_empty_pos()
        self.goal_pos = self._get_random_empty_pos()
        while np.array_equal(self.agent_pos, self.goal_pos) or np.linalg.norm(self.agent_pos - self.goal_pos) < 20:
            self.goal_pos = self._get_random_empty_pos()
            
        self.previous_distance = np.linalg.norm(self.agent_pos - self.goal_pos)
        
        if self.add_dynamic_ob:
            self.dy_ob_pos = np.array([40, 40])
            self.dy_ob_direction = 1
        return self._get_obs(), {}

    def _get_random_empty_pos(self):
        while True:
            x = np.random.randint(0, self.map_size)
            y = np.random.randint(0, self.map_size)
            if self.global_map[x, y] == 0:
                if self.add_dynamic_ob:
                    # 避开中央的高频巡逻区
                    if not (35 < x < 45 and 20 < y < 60):
                        return np.array([x, y])
                return np.array([x, y])

    def _get_obs(self):
        local_grid = np.zeros((7, 7), dtype=np.float32) 
        ax, ay = self.agent_pos
        temp_view = np.where(self.global_map == 1, 1.0, 0.0).astype(np.float32)
        
        if self.add_dynamic_ob:
            tox, toy = self.dy_ob_pos
            half_s = self.dy_ob_size // 2
            xmin, xmax = max(0, tox - half_s), min(self.map_size, tox + half_s + 1)
            ymin, ymax = max(0, toy - half_s), min(self.map_size, toy + half_s + 1)
            temp_view[xmin:xmax, ymin:ymax] = 0.5

        for i in range(7):
            for j in range(7):
                gx = ax + i - 3
                gy = ay + j - 3
                if 0 <= gx < self.map_size and 0 <= gy < self.map_size:
                    local_grid[i, j] = temp_view[gx, gy]
                else:
                    local_grid[i, j] = 1.0 
                    
        target_vector = (self.goal_pos - self.agent_pos).astype(np.float32)
        dist_to_goal = np.linalg.norm(target_vector)
        if dist_to_goal > 0:
            target_vector = target_vector / dist_to_goal
            
        return {"local_grid": local_grid, "target_vector": target_vector}

    def step(self, action):
        if hasattr(action, 'item'): action = int(action.item())
        else: action = int(action)
        self.current_step += 1
        
        # ====== 修复：让巡逻车横穿对角线 (Y轴 20 到 60 之间) ======
        if self.add_dynamic_ob:
            self.dy_ob_pos[1] += self.dy_ob_speed * self.dy_ob_direction
            if self.dy_ob_pos[1] <= 20 or self.dy_ob_pos[1] >= 60:
                self.dy_ob_direction *= -1

        dx, dy = self.action_mapping[action]
        new_pos = self.agent_pos + np.array([dx, dy])
        
        reward = 0.0
        terminated = False
        truncated = False
        
        hit_dy_ob = False
        if self.add_dynamic_ob:
            tox, toy = self.dy_ob_pos
            half_s = self.dy_ob_size // 2
            if tox - half_s - 1 < new_pos[0] < tox + half_s + 1 and \
               toy - half_s - 1 < new_pos[1] < toy + half_s + 1:
                hit_dy_ob = True

        # 环境内部判断碰撞并给予惩罚
        hit_wall = (new_pos[0] < 0 or new_pos[0] >= self.map_size or
                    new_pos[1] < 0 or new_pos[1] >= self.map_size or
                    self.global_map[new_pos[0], new_pos[1]] == 1)

        if hit_wall or hit_dy_ob:
            reward = -10.0 if hit_dy_ob else -5.0
        else:
            self.agent_pos = new_pos
            current_distance = np.linalg.norm(self.agent_pos - self.goal_pos)
            reward += (self.previous_distance - current_distance) * 2.0
            self.previous_distance = current_distance
            reward -= 0.1 

        # 终点检测
        dist_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        if dist_to_goal <= 1.5:
            reward += 100.0
            terminated = True
            
        if self.current_step >= self.max_steps:
            truncated = True
            
        # 把碰撞信息传出去给可视化脚本用
        info = {'hit_wall': hit_wall, 'hit_dy_ob': hit_dy_ob}
        return self._get_obs(), reward, terminated, truncated, info
