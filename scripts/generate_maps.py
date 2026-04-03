import numpy as np
import os
import math
import sys

# 将工作目录切换到项目根目录，确保路径正确
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def generate_map(level):
    """
    复现 Level 1 到 Level 5 的动态随机栅格地图生成逻辑
    """
    if level <= 3:
        map_size = 40
        num_blocks = 30 + level * 25  # L1=55, L2=80, L3=105
    else:
        map_size = 80
        num_blocks = 150 + (level - 4) * 100  # L4=150, L5=250

    map_grid = np.zeros((map_size, map_size), dtype=int)

    # 固定随机种子，确保与难度等级绑定，每次生成的地图一样
    np.random.seed(level * 100)

    for _ in range(num_blocks):
        # 随机生成障碍物的宽和高 (0 到 3，实际占用 1x1 到 4x4)
        w = int(np.floor(np.random.rand() * 4))
        h = int(np.floor(np.random.rand() * 4))

        # 随机生成障碍物的左上角坐标 (预留边界，防止越界)
        x = int(np.floor(np.random.rand() * (map_size - w - 2))) + 1
        y = int(np.floor(np.random.rand() * (map_size - h - 2))) + 1

        # 填充障碍物区块
        map_grid[x:x+w+1, y:y+h+1] = 1

    # 强制挖空起点和终点周围的安全区域 (10% 比例)
    safe_zone = math.ceil(map_size * 0.1)
    map_grid[0:safe_zone, 0:safe_zone] = 0
    map_grid[map_size-safe_zone:map_size, map_size-safe_zone:map_size] = 0

    # 加上一圈最外围的墙壁边界 (封闭环境)
    map_grid[0, :] = 1
    map_grid[-1, :] = 1
    map_grid[:, 0] = 1
    map_grid[:, -1] = 1
    
    # 重新挖空真正的绝对起点 [1, 1] 和终点 [map_size-2, map_size-2]
    map_grid[1, 1] = 0
    map_grid[map_size-2, map_size-2] = 0

    return map_grid

def main():
    save_dir = os.path.join(project_root, "data", "csv_maps")
    os.makedirs(save_dir, exist_ok=True)
    
    print("开始生成训练基准地图...")
    for level in range(1, 6):
        map_grid = generate_map(level)
        file_name = f"Phase1_Map_Level_{level}.csv"
        file_path = os.path.join(save_dir, file_name)
        
        # 导出为 CSV，使用整数格式
        np.savetxt(file_path, map_grid, delimiter=',', fmt='%d')
        print(f"✅ 成功导出: {file_name} (尺寸: {map_grid.shape[0]}x{map_grid.shape[1]})")
        
    print(f"\n全部 5 张地图已保存至: {save_dir}")

if __name__ == "__main__":
    main()
