# scripts/exp4_llm_hallucination_v2.py
# 重构版：真正测试「空间幻觉」而非「格式幻觉」
import re
import random
import json
from openai import OpenAI

# ================= 配置区 =================
CURRENT_MODEL = "qwen2.5-coder-1.5b-instruct"   # 切换模型时改这里
TEST_ROUNDS   = 50
GRID_H, GRID_W = 5, 7           # 雷达图尺寸（行 x 列）
OBSTACLE_RATIO = 0.25           # 随机障碍物密度
# ==========================================

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")


# ──────────────────────────────────────────
# 地图生成器
# ──────────────────────────────────────────
def generate_random_scene():
    """
    随机生成一张雷达图，返回：
      grid       : 2D list，值为 'X'/'.'/'R'
      robot_pos  : (row, col)
      free_cells : list of (row, col)  所有空地
      shortlist  : list of [col, row]  候选安全点（含陷阱）
      target_dx  : int  终点相对偏移
      target_dy  : int
    """
    while True:
        grid = [['.' for _ in range(GRID_W)] for _ in range(GRID_H)]

        # 随机放障碍物
        for r in range(GRID_H):
            for c in range(GRID_W):
                if random.random() < OBSTACLE_RATIO:
                    grid[r][c] = 'X'

        # 随机放机器人（必须在空地）
        free = [(r, c) for r in range(GRID_H) for c in range(GRID_W) if grid[r][c] == '.']
        if len(free) < 4:
            continue  # 空地太少，重新生成
        robot_pos = random.choice(free)
        grid[robot_pos[0]][robot_pos[1]] = 'R'

        # 重新收集空地（排除机器人位置）
        free_cells = [(r, c) for r in range(GRID_H) for c in range(GRID_W) if grid[r][c] == '.']
        if len(free_cells) < 3:
            continue

        # 生成候选点 shortlist（混入 1-2 个陷阱：障碍物坐标）
        wall_cells = [(r, c) for r in range(GRID_H) for c in range(GRID_W) if grid[r][c] == 'X']
        good_candidates = random.sample(free_cells, min(3, len(free_cells)))
        trap_candidates = random.sample(wall_cells, min(random.randint(1, 2), len(wall_cells)))
        all_candidates  = good_candidates + trap_candidates
        random.shuffle(all_candidates)

        # 转成 [col, row] 格式（与原脚本一致）
        shortlist = [[c, r] for r, c in all_candidates]

        target_dx = random.randint(3, 15) * random.choice([-1, 1])
        target_dy = random.randint(3, 15) * random.choice([-1, 1])

        return grid, robot_pos, free_cells, shortlist, wall_cells, target_dx, target_dy


def grid_to_ascii(grid):
    return '\n'.join(' '.join(row) for row in grid)


def build_prompt(grid, shortlist, target_dx, target_dy):
    ascii_map = grid_to_ascii(grid)
    candidates_str = ', '.join(str(s) for s in shortlist)
    return f"""雷达图(R=机器人, .=可走空地, X=墙壁, -=禁止反向):
{ascii_map}

终点在 dx={target_dx}, dy={target_dy}。
机器人已卡住。候选局部坐标（列,行）有：{candidates_str}。
请结合雷达图，严格从候选坐标中挑选一个可走的空地！只输出类似 [2, 3] 的坐标，不要任何解释！"""


# ──────────────────────────────────────────
# 主测试逻辑
# ──────────────────────────────────────────
def test_hallucination():
    print(f"🚀 开始测试 {CURRENT_MODEL} 的空间幻觉率 (共 {TEST_ROUNDS} 轮，每轮随机地图)...\n")

    stats = {
        "format_fail":   0,   # 输出格式不合规
        "oob":           0,   # 坐标越界（超出地图范围）
        "wall_pick":     0,   # 选了墙壁（空间幻觉！）
        "not_in_list":   0,   # 选了不在候选列表里的点
        "perfect":       0,   # 完全正确：格式✓ + 在列表✓ + 是空地✓
    }

    records = []  # 用于最后详细打印

    for i in range(TEST_ROUNDS):
        grid, robot_pos, free_cells, shortlist, wall_cells, dx, dy = generate_random_scene()
        prompt = build_prompt(grid, shortlist, dx, dy)

        try:
            response = client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": "你只能输出形如 [X, Y] 的坐标，不要任何额外文字。"},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.8,
                max_tokens=20
            )
            reply = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [{i+1:02d}] ⚠️  API 异常: {e}")
            stats["format_fail"] += 1
            continue

        # ── 检查 1：格式 ──
        m = re.fullmatch(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', reply)
        if not m:
            stats["format_fail"] += 1
            label = "❌ 格式错误"
            records.append((i+1, label, reply, shortlist, None))
            print(f"  [{i+1:02d}] {label} -> '{reply}'")
            continue

        col, row = int(m.group(1)), int(m.group(2))

        # ── 检查 2：越界 ──
        if not (0 <= row < GRID_H and 0 <= col < GRID_W):
            stats["oob"] += 1
            label = "❌ 坐标越界"
            records.append((i+1, label, reply, shortlist, (col, row)))
            print(f"  [{i+1:02d}] {label} -> {reply}  (地图大小 {GRID_H}x{GRID_W})")
            continue

        # ── 检查 3：是否在候选列表 ──
        in_list = [col, row] in shortlist
        if not in_list:
            stats["not_in_list"] += 1
            label = "⚠️  不在候选列表"
            records.append((i+1, label, reply, shortlist, (col, row)))
            print(f"  [{i+1:02d}] {label} -> {reply}  候选={shortlist}")
            continue

        # ── 检查 4：是否选了墙壁（核心空间幻觉检测）──
        cell_value = grid[row][col]
        if cell_value == 'X':
            stats["wall_pick"] += 1
            label = "🚨 空间幻觉！选了墙壁"
            records.append((i+1, label, reply, shortlist, (col, row)))
            print(f"  [{i+1:02d}] {label} -> {reply}")
            # 打印地图帮助调试
            print(f"       地图:\n" + '\n'.join('       ' + ' '.join(r) for r in grid))
        else:
            stats["perfect"] += 1
            label = "✅ 完全正确"
            records.append((i+1, label, reply, shortlist, (col, row)))
            print(f"  [{i+1:02d}] {label} -> {reply}")

    # ──────────────────────────────────────────
    # 汇总报告
    # ──────────────────────────────────────────
    print("\n" + "="*50)
    print(f"📊 模型: {CURRENT_MODEL}   测试轮数: {TEST_ROUNDS}")
    print("="*50)
    print(f"  ✅ 完全正确          : {stats['perfect']:3d} / {TEST_ROUNDS}  ({stats['perfect']/TEST_ROUNDS*100:.1f}%)")
    print(f"  ❌ 格式错误          : {stats['format_fail']:3d} / {TEST_ROUNDS}  ({stats['format_fail']/TEST_ROUNDS*100:.1f}%)")
    print(f"  ❌ 坐标越界          : {stats['oob']:3d} / {TEST_ROUNDS}  ({stats['oob']/TEST_ROUNDS*100:.1f}%)")
    print(f"  ⚠️  不在候选列表     : {stats['not_in_list']:3d} / {TEST_ROUNDS}  ({stats['not_in_list']/TEST_ROUNDS*100:.1f}%)")
    print(f"  🚨 空间幻觉(选墙壁)  : {stats['wall_pick']:3d} / {TEST_ROUNDS}  ({stats['wall_pick']/TEST_ROUNDS*100:.1f}%)")
    print("="*50)

    spatial_hallucination_rate = (stats['wall_pick'] / TEST_ROUNDS) * 100
    format_compliance_rate     = ((TEST_ROUNDS - stats['format_fail']) / TEST_ROUNDS) * 100
    print(f"\n  🏆 格式遵从率        : {format_compliance_rate:.1f}%")
    print(f"  🧠 空间幻觉率        : {spatial_hallucination_rate:.1f}%  ← 论文核心指标")
    print("="*50)

    # 保存结果到 JSON（方便跨模型对比）
    output = {
        "model": CURRENT_MODEL,
        "test_rounds": TEST_ROUNDS,
        "stats": stats,
        "format_compliance_rate": round(format_compliance_rate, 1),
        "spatial_hallucination_rate": round(spatial_hallucination_rate, 1),
    }
    out_path = f"results_hallucination_{CURRENT_MODEL.replace('/', '_')}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 结果已保存至 {out_path}")


if __name__ == "__main__":
    test_hallucination()
