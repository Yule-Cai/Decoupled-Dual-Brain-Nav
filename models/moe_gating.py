# models/moe_gating.py
import numpy as np
from models.llm_parser import LLMNavigator

class MoEArbiter:
    # ========== 【核心优化：耐心值增加到 15 步】 ==========
    def __init__(self, rl_model, patience=15): 
        self.rl_model = rl_model
        self.llm_nav = LLMNavigator()
        self.patience = patience
        self.pos_history = []
        self.mode = "RL"
        self.current_waypoint = None
        self.waypoint_steps = 0
        self.max_waypoint_steps = 25 
        
        self.trap_zones = [] 
        self.trap_radius = 8.0 
        
        self.history_wps = [] 
        self.intervention_logs = []

    def predict(self, obs_flat, agent_pos, goal_pos, global_map):
        self.pos_history.append(agent_pos.copy())
        if len(self.pos_history) > self.patience:
            self.pos_history.pop(0)

        local_grid = obs_flat[:49]

        if self.mode == "LLM_WAYPOINT":
            self.waypoint_steps += 1
            dist_to_wp = np.linalg.norm(agent_pos - self.current_waypoint)
            
            if dist_to_wp <= 1.5 or self.waypoint_steps > self.max_waypoint_steps:
                self.mode = "RL"
                self.current_waypoint = None
                self.pos_history.clear() 
                return self._get_rl_action_with_tabu(obs_flat, agent_pos, goal_pos)
            else:
                fake_target_vec = (self.current_waypoint - agent_pos).astype(np.float32)
                dist = np.linalg.norm(fake_target_vec)
                if dist > 0: fake_target_vec /= dist
                fake_obs = np.concatenate([local_grid, fake_target_vec])
                
                action, _ = self.rl_model.predict(fake_obs, deterministic=True)
                if hasattr(action, 'item'): action = int(action.item())
                else: action = int(action)
                return action, f"Waypoint"

        if self.mode == "RL":
            dist_to_goal = np.linalg.norm(agent_pos - goal_pos)
            if dist_to_goal >= 5.0 and len(self.pos_history) == self.patience:
                dist_moved = np.linalg.norm(self.pos_history[-1] - self.pos_history[0])
                
                # ========== 【核心优化：极其严格的卡死判定】 ==========
                # 给了 15 步的耐心，如果 15 步下来位移还不到 1.5 格，那才是真卡死！
                if dist_moved < 1.5:
                    print(f"\n🚨 [仲裁器] 陷入死胡同！打下思想钢印，呼叫 LLM 强制推进...")
                    self.mode = "LLM_WAYPOINT"
                    self.waypoint_steps = 0
                    
                    self.trap_zones.append(agent_pos.copy())
                    
                    try:
                        new_wp = self.llm_nav.get_waypoint(global_map, agent_pos, goal_pos, self.history_wps)
                    except TypeError:
                        new_wp = self.llm_nav.get_waypoint(global_map, agent_pos, goal_pos)
                        
                    self.current_waypoint = new_wp
                    self.history_wps.append(new_wp.copy())
                    self.intervention_logs.append((agent_pos.copy(), new_wp.copy()))
                    self.pos_history.clear()
                    
                    fake_target_vec = (self.current_waypoint - agent_pos).astype(np.float32)
                    dist = np.linalg.norm(fake_target_vec)
                    if dist > 0: fake_target_vec /= dist
                    fake_obs = np.concatenate([local_grid, fake_target_vec])
                    
                    action, _ = self.rl_model.predict(fake_obs, deterministic=True)
                    if hasattr(action, 'item'): action = int(action.item())
                    else: action = int(action)
                    return action, f"Waypoint"

        return self._get_rl_action_with_tabu(obs_flat, agent_pos, goal_pos)

    def _get_rl_action_with_tabu(self, obs_flat, agent_pos, goal_pos):
        if not self.trap_zones:
            action, _ = self.rl_model.predict(obs_flat, deterministic=True)
            if hasattr(action, 'item'): return int(action.item()), "RL"
            return int(action), "RL"
            
        base_target_vec = (goal_pos - agent_pos).astype(np.float32)
        repulsion_vec = np.array([0.0, 0.0], dtype=np.float32)
        
        for trap in self.trap_zones:
            dist_to_trap = np.linalg.norm(agent_pos - trap)
            if dist_to_trap < self.trap_radius:
                force_magnitude = (self.trap_radius - dist_to_trap) * 2.0 
                repulsion_vec += (agent_pos - trap) * force_magnitude
                
        final_vec = base_target_vec + repulsion_vec
        dist = np.linalg.norm(final_vec)
        if dist > 0: final_vec /= dist 
        
        fake_obs = np.concatenate([obs_flat[:49], final_vec])
        action, _ = self.rl_model.predict(fake_obs, deterministic=True)
        if hasattr(action, 'item'): return int(action.item()), "RL"
        return int(action), "RL"
