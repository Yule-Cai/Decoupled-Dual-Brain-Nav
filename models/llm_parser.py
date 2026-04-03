# models/llm_parser.py
from openai import OpenAI
import numpy as np
import re

class LLMNavigator:
    def __init__(self, base_url="http://127.0.0.1:1234/v1", api_key="lm-studio"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _has_line_of_sight(self, grid, x0, y0, x1, y1):
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if grid[x, y] == 1: return False
                err -= dy
                if err < 0: y += sy; err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if grid[x, y] == 1: return False
                err -= dx
                if err < 0: x += sx; err += dy
                y += sy
        return grid[x, y] == 0

    def get_waypoint(self, global_map, agent_pos, goal_pos, history_wps=None, view_radius=8):
        if history_wps is None: history_wps = []
        
        map_size = global_map.shape[0]
        ax, ay = agent_pos
        min_x, max_x = max(0, ax - view_radius), min(map_size, ax + view_radius + 1)
        min_y, max_y = max(0, ay - view_radius), min(map_size, ay + view_radius + 1)
        
        local_view = global_map[min_x:max_x, min_y:max_y]
        rel_ax, rel_ay = ax - min_x, ay - min_y
        
        dx_to_goal = goal_pos[0] - ax
        dy_to_goal = goal_pos[1] - ay
        
        grid_str = ""
        valid_waypoints = [] 
        
        for i in range(local_view.shape[0]):
            for j in range(local_view.shape[1]):
                if i == rel_ax and j == rel_ay:
                    grid_str += "R " 
                elif local_view[i, j] == 1:
                    grid_str += "X " 
                else:
                    global_i, global_j = i + min_x, j + min_y
                    dx_to_wp = global_i - ax
                    dy_to_wp = global_j - ay
                    dot_product = dx_to_goal * dx_to_wp + dy_to_goal * dy_to_wp
                    
                    if dot_product >= 0 and (abs(i - rel_ax) + abs(j - rel_ay) >= 3):
                        if self._has_line_of_sight(local_view, rel_ax, rel_ay, i, j):
                            grid_str += ". " 
                            valid_waypoints.append((i, j))
                            continue
                    
                    grid_str += "- " 
            grid_str += "\n"

        if not valid_waypoints:
             return agent_pos + np.array([np.random.randint(-1, 2), np.random.randint(-1, 2)])

        # ========== 【你的绝杀 1：提前计算兜底并排序，用于候选清单和异常抛出】 ==========
        def sort_key(p):
            global_p = np.array([p[0]+min_x, p[1]+min_y])
            dist_goal = np.linalg.norm(global_p - goal_pos)
            history_penalty = 0
            for hw in history_wps:
                dist_hw = np.linalg.norm(global_p - hw)
                if dist_hw < 8.0:
                    history_penalty += (8.0 - dist_hw) * 3.0
            return dist_goal + history_penalty

        valid_waypoints.sort(key=sort_key)
        
        # 提取前 5 个最优秀的候选坐标，明明白白地喂给小模型
        top_k = valid_waypoints[:5]
        top_k_strs = [f"[{p[0]+min_x}, {p[1]+min_y}]" for p in top_k]
        candidates_str = ", ".join(top_k_strs)

        # ========== 【你的绝杀 2：降低 LLM 认知负担的 Prompt】 ==========
        prompt = f"雷达图(R=机器,.=可选空地,X=墙,-=禁止反向):\n{grid_str}\n终点在 dx={dx_to_goal}, dy={dy_to_goal}。\n机器人卡住了。我为你筛选了几个绝对安全的候选坐标：{candidates_str}。\n请结合雷达图，从这些坐标中严格挑选一个最平滑的！只输出类似 [2, 5] 的坐标！"

        try:
            response = self.client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": "你只能输出形如 [X, Y] 的内容，严禁输出候选列表以外的坐标。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.01, 
                max_tokens=10
            )
            reply = response.choices[0].message.content.strip()
            print(f"   [LLM 思考] : {reply}")
            
            match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', reply)
            if match:
                rel_wx, rel_wy = int(match.group(1)) - min_x, int(match.group(2)) - min_y
                if (rel_wx, rel_wy) in valid_waypoints:
                    global_wp = np.array([rel_wx + min_x, rel_wy + min_y])
                    if not any(np.linalg.norm(global_wp - hw) < 3.0 for hw in history_wps):
                        return global_wp
                        
            # 如果大模型依然犯病没选对，直接用榜单第一名
            return np.array([valid_waypoints[0][0] + min_x, valid_waypoints[0][1] + min_y])
            
        except Exception as e:
            # ========== 【你的绝杀 3：修复死亡异常返回】 ==========
            print(f"   ⚠️ [LLM 异常/超时] 直接启动底层兜底策略！")
            return np.array([valid_waypoints[0][0] + min_x, valid_waypoints[0][1] + min_y])
