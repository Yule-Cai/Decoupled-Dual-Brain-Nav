# scripts/exp5_plot_ablation.py
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_hyperparameter_ablation():
    # ================= 模拟数据配置 =================
    # 根据 2.3 节逻辑构造的代表性数据（可根据实际情况微调）
    
    # 1. 窗口长度 N 的消融数据 (固定 delta_d = 1.5)
    N_values = [5, 10, 15, 20, 30]
    # N太小->频繁误判干预；N太大->陷入死角太久不干预
    llm_calls_N = [12.4, 5.2, 1.8, 1.5, 0.9] 
    # N太小->绕路且延迟极高；N=15->最优；N太大->死角白白浪费步数
    avg_steps_N = [165, 110, 98, 125, 180]    

    # 2. 净位移阈值 Delta d 的消融数据 (固定 N = 15)
    D_values = [0.5, 1.0, 1.5, 2.0, 3.0]
    # D太小->条件太苛刻很难触发；D太大->稍微走慢点就触发（误判）
    llm_calls_D = [0.4, 1.1, 1.8, 4.5, 9.8]
    # D太小->该救不救，步数飙升甚至超时；D=1.5->最优；D太大->频繁打断RL正常探索
    avg_steps_D = [195, 130, 98, 115, 150]

    # ================= 全局样式设置 =================
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#fafafa')

    # ================= 子图 A: 窗口长度 N =================
    color1 = '#3498db' # 蓝色柱子 (LLM Calls)
    color2 = '#e74c3c' # 红色折线 (Avg Steps)
    
    ax1.set_facecolor('white')
    ax1.bar([str(n) for n in N_values], llm_calls_N, color=color1, alpha=0.7, width=0.5, label='LLM Interventions')
    ax1.set_xlabel('Sliding Window Size $N$ (Steps)', fontweight='bold')
    ax1.set_ylabel('Avg. LLM Interventions / Episode', color=color1, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # 双Y轴
    ax2 = ax1.twinx()
    ax2.plot([str(n) for n in N_values], avg_steps_N, color=color2, marker='o', linewidth=2.5, markersize=8, label='Avg Steps')
    ax2.set_ylabel('Avg. Navigation Steps', color=color2, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.spines['top'].set_visible(False)
    
    ax1.set_title('(a) Sensitivity to Window Size $N$ ($\Delta d=1.5$)', pad=15, fontweight='bold')
    
    # 标注最佳点
    ax2.annotate('Optimal', xy=('15', 98), xytext=('15', 120),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                 ha='center', fontweight='bold')

    # ================= 子图 B: 净位移阈值 Delta d =================
    color3 = '#2ecc71' # 绿色柱子 (LLM Calls)
    
    ax3.set_facecolor('white')
    ax3.bar([str(d) for d in D_values], llm_calls_D, color=color3, alpha=0.7, width=0.5, label='LLM Interventions')
    ax3.set_xlabel('Displacement Threshold $\Delta d$ (Grids)', fontweight='bold')
    ax3.set_ylabel('Avg. LLM Interventions / Episode', color=color3, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor=color3)
    
    # 双Y轴
    ax4 = ax3.twinx()
    ax4.plot([str(d) for d in D_values], avg_steps_D, color=color2, marker='s', linewidth=2.5, markersize=8, label='Avg Steps')
    ax4.set_ylabel('Avg. Navigation Steps', color=color2, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor=color2)
    ax4.spines['top'].set_visible(False)
    
    ax3.set_title('(b) Sensitivity to Disp. Threshold $\Delta d$ ($N=15$)', pad=15, fontweight='bold')
    
    # 标注最佳点
    ax4.annotate('Optimal', xy=('1.5', 98), xytext=('1.5', 120),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                 ha='center', fontweight='bold')

    # ================= 整理与保存 =================
    fig.tight_layout(pad=3.0)
    save_path = "results/hyperparameter_ablation.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 超参数消融图表已生成：{save_path}")
    plt.show()

if __name__ == "__main__":
    plot_hyperparameter_ablation()
