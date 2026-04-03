# scripts/analyze_and_plot.py
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import glob

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(project_root, "benchmark_results")

# 配置高大上的学术主题
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12

# 名字美化字典：把杂乱的文件名变成论文里的标准称呼
NAME_MAPPING = {
    "Gemma-3-1B": "Gemma-3 (1B)",
    "qwen2.5-coder-1.5b-instruct": "Qwen2.5-Coder (1.5B)",
    "LFM-2.5-1.2B": "LFM-2.5 (1.2B)"
}

# 黄金铁三角专属配色
COLOR_MAPPING = {
    "Gemma-3 (1B)": "#4285F4",         # Google 蓝
    "Qwen2.5-Coder (1.5B)": "#00C78C", # 极客绿
    "LFM-2.5 (1.2B)": "#FF6D00"        # 活力橙
}

def load_all_data():
    json_files = glob.glob(os.path.join(RESULT_DIR, "*_data.json"))
    if not json_files:
        print(f"❌ 警告: 在 {RESULT_DIR} 下没有找到任何数据文件！")
        return None, []
        
    processed_data = []
    models_found = []
    
    for path in json_files:
        filename = os.path.basename(path)
        raw_model_name = filename.replace("_data.json", "")
        # 美化模型名字
        pretty_name = NAME_MAPPING.get(raw_model_name, raw_model_name)
        if pretty_name not in models_found:
            models_found.append(pretty_name)
        
        with open(path, 'r') as f:
            data = json.load(f)
            
        for map_name, runs in data.items():
            map_id = map_name.split('_')[1]
            for r in runs:
                # 计算纯粹的 LLM 平均每次推理耗时
                # (排除掉完全没有调用 LLM 的回合)
                if r['llm_calls'] > 0:
                    # 减去大约 0.01 秒的物理仿真基础时间，除以调用次数
                    latency_per_call = max(0, r['total_time'] - 0.01) / r['llm_calls']
                else:
                    latency_per_call = 0
                    
                processed_data.append({
                    "Model": pretty_name,
                    "Map": f"Map {map_id}",
                    "Success": 1.0 if r['success'] else 0.0,
                    "Steps": r['steps'] if r['success'] else None, 
                    "LLM_Calls": r['llm_calls'],
                    "Latency": latency_per_call
                })
    return pd.DataFrame(processed_data), models_found

def plot_benchmark(df, models):
    # 如果找到了不在预设里的模型，分配默认颜色
    default_colors = ["#9C27B0", "#E91E63", "#3F51B5"]
    model_colors = {}
    for i, m in enumerate(models):
        model_colors[m] = COLOR_MAPPING.get(m, default_colors[i % len(default_colors)])

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('Empirical Study: 1B-Class Lightweight LLMs in Embodied Navigation', fontsize=20, fontweight='bold', y=0.98)

    # 1. 成功率对比 (Bar Plot)
    sr_df = df.groupby(['Model', 'Map'])['Success'].mean().reset_index()
    sns.barplot(x='Map', y='Success', hue='Model', data=sr_df, palette=model_colors, ax=axes[0, 0])
    axes[0, 0].set_title('A. Navigation Success Rate (SR) ↑', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_ylim(0, 1.1)
    axes[0, 0].legend(title='Agent Backend', loc='lower right')

    # 2. 平均步数对比 (Box Plot, 仅统计成功的路线)
    steps_df = df[df['Success'] == 1.0]
    sns.boxplot(x='Map', y='Steps', hue='Model', data=steps_df, palette=model_colors, ax=axes[0, 1], showfliers=False)
    axes[0, 1].set_title('B. Path Efficiency (Steps to Goal) ↓', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].legend(title='Agent Backend', loc='upper right')

    # 3. LLM 呼叫次数 (Bar Plot)
    sns.barplot(x='Map', y='LLM_Calls', hue='Model', data=df, palette=model_colors, ax=axes[1, 0])
    axes[1, 0].set_title('C. LLM Intervention Frequency ↓', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Mean Calls per Episode')
    axes[1, 0].legend(title='Agent Backend', loc='upper right')

    # 4. LLM 单次推理延迟对比 (Bar Plot)
    latency_df = df[df['Latency'] > 0].groupby(['Model'])['Latency'].mean().reset_index()
    sns.barplot(x='Model', y='Latency', data=latency_df, palette=model_colors, ax=axes[1, 1])
    axes[1, 1].set_title('D. Mean LLM Inference Latency ↓', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Time per Call (Seconds)')
    
    # 在柱状图上标注具体秒数
    for p in axes[1, 1].patches:
        axes[1, 1].annotate(f'{p.get_height():.2f}s', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(RESULT_DIR, "lightweight_llms_ablation.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 完美！三大轻量级模型消融对比图已保存至: {save_path}")
    plt.show()

def main():
    combined_df, models_found = load_all_data()
    
    if combined_df is not None and not combined_df.empty:
        print(f"✅ 成功加载以下模型的数据: {', '.join(models_found)}")
        plot_benchmark(combined_df, models_found)
    else:
        print("❌ 未找到有效数据，请检查 benchmark_results 文件夹。")

if __name__ == "__main__":
    main()
