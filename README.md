```markdown
# 🧠 Decoupled Dual-Brain Embodied Navigation (RL + LLM)
> **Empirical Evaluation of RL+LLM for Navigation under Strict Partial Observability**

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)
![Stable Baselines3](https://img.shields.io/badge/SB3-RL-brightgreen)
![Edge Computing](https://img.shields.io/badge/Hardware-Orange_Pi_4-ff7f0e)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[📝 Paper (Coming Soon)]** | **[🎬 Video Demo]** | **[📦 Dataset]**

</div>

---

<p align="center">
  <em>An ultra-robust, fault-tolerant navigation framework that physically decouples high-frequency muscle memory (RL) from low-frequency cognitive reasoning (LLM), achieving a 100% success rate in blind mode and surviving up to ~77 seconds of cognitive freeze on pure-CPU edge devices.</em>
</p>

---

## 📢 News
- **[202X.0X]** 🎉 Phase 1 (2D Simulation & Edge Deployment) is officially released!
- **[202X.0X]** 🚀 Deployed on Orange Pi 4 (pure ARM CPU) with LFM-2.5-1.2B (4-bit GGUF).
- **[Future]** 🚧 Phase 2 (3D Continuous Physics in ROS 2 / Gazebo) is currently under active development.

---

## ✨ Key Features
- **🛡️ Absolute Physical Safety (0s Fallback):** Unlike end-to-end VLMs, our decoupled PPO agent maintains high-frequency local collision repulsion even if the LLM crashes or suffers from severe I/O latency.
- **🕵️ Black-Box Spatio-Temporal Arbiter:** A completely model-agnostic trigger mechanism ($N=15$ steps, $\Delta d < 1.5$ grids) that perfectly detects "implicit deadlocks" where traditional entropy-based triggers suffer from "blind confidence."
- **🧩 ASCII Topology for Code-LLMs:** Eliminates the need for heavy Vision-Language Models. We serialize local radar grids into lightweight ASCII matrices, unlocking the latent spatial reasoning power of Code-pretrained models (e.g., *Qwen2.5-Coder*).
- **📱 Edge-Ready (SWaP-C Constrained):** Successfully runs zero-shot spatial reasoning using 1B-class models on a $35 Single-Board Computer (Orange Pi 4) without any GPU acceleration.

---

## 🏗 System Architecture

The core philosophy of this framework is: **"RL handles the immediate survival, while LLM handles the macroscopic escape."**

<div align="center">
  <img src="docs/architecture_diagram.jpg" alt="System Architecture" width="85%">
</div>

1. The **RL Base (Cerebellum)** operates at high frequency based on local radar grids (Blind Mode).
2. The **Arbiter** monitors the trajectory. If a deadlock is detected, it freezes the macroscopic goal and requests help.
3. The **LLM (Cerebrum)** receives the ASCII topology, processes the spatial constraints, and outputs a strategic escape waypoint to pull the RL agent out of the trap.

---

## 📊 Experimental Results & Insights

Our extensive Phase 1 evaluations across L1-L5 maps yield several profound engineering insights:

### 1. The "Instruction Compliance Paradox" (LLM Ablation)
We designed an **Adversarial Spatial Grounding Test** where candidate waypoints were intentionally contaminated with wall coordinates.

<div align="center">
  <img src="docs/hallucination_benchmark.jpg" alt="Hallucination Benchmark" width="85%">
</div>

- **Qwen2.5-Coder (1.5B):** Achieved the highest formatting compliance but suffered a **21% spatial hallucination rate**. Being highly "instruction-tuned" makes it a perfect text executor but a vulnerable physical gatekeeper.
- **LFM-2.5 (1.2B):** Exhibited a "safety via rebellion" paradox. It ignored the flawed external shortlist **83%** of the time, relying on its internal spatial prior to find safe spots, dropping the hallucination rate to just **7%**. The remaining errors were flawlessly absorbed by the RL's collision repulsion.

### 2. Arbiter Hyperparameter Tuning (Pareto Optimal)
<div align="center">
  <img src="docs/ablation_L2_vs_L5_final.jpg" alt="Hyperparameter Ablation" width="85%">
</div>

Ablation studies on $N$ (Sliding Window) and $\Delta d$ (Displacement Threshold) confirmed that **$N=15$ and $\Delta d=1.5$** is the universal optimal configuration, balancing intervention efficiency and deadlock-breaking robustness across both trap-heavy (L2) and maze-like (L5) environments.

### 3. Extreme Edge Latency Tolerance (Orange Pi 4)
We deployed the full framework on an **Orange Pi 4 (ARM aarch64, 6-core, 3.8GB RAM, CPU-only)** running `LFM-2.5-1.2B` in 4-bit GGUF format on the L5 Hard map.

| Hardware/System Metrics | Orange Pi 4 Edge Deployment Results |
| :----------------------- | :----------------------------------- |
| **Terminal Success Rate** | **90.0%** (9/10 Episodes) |
| **Avg. LLM Latency** | **37.82 s** / call |
| **Peak Latency Tolerance**| **77.43 s** 👑 *(Surviving severe cognitive freeze without crashing)* |
| **Minimum Fallback** | **0.00 s** *(Instant Tabu Search recovery)* |

---

## 🚀 Getting Started

### 1. Prerequisites & Installation
We recommend using Conda to manage your environment:
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Decoupled-Dual-Brain-Nav.git
cd Decoupled-Dual-Brain-Nav

# Create and activate conda environment
conda create -n dualbrain python=3.10 -y
conda activate dualbrain

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Benchmark
To reproduce the L1-L5 baseline benchmark (APF, A*, Pure PPO, Ours):
```bash
python scripts/exp1_consolidated_benchmark.py
```

### 3. Evaluate LLM Hallucinations
Launch your local LLM server (e.g., LM Studio / Ollama) on `http://127.0.0.1:1234/v1` and run:
```bash
python scripts/exp4_llm_hallucination.py
```

---

## 📂 Repository Structure
```text
Decoupled-Dual-Brain-Nav/
├── data/
│   └── csv_maps/            # L1-L5 procedural generation maps
├── docs/                    # High-res diagrams and plots
├── envs/
│   └── grid_nav_env.py      # Core RL environment (Strict Blind Mode)
├── models/
│   ├── moe_gating.py        # The Spatio-Temporal Arbiter & Dual-Brain logic
│   └── saved_weights/       # Pre-trained PPO weights (.zip)
├── scripts/
│   ├── generate_maps.py           # Map synthesis
│   ├── train_curriculum.py        # 6M-step Curriculum Learning pipeline
│   ├── exp1_consolidated_benchmark.py # Multi-algorithm evaluation
│   ├── exp4_llm_hallucination.py  # Adversarial testing
│   ├── exp5_ablation_real.py      # Parameter sensitivity
│   └── exp6_edge_hardware_test.py # Orange Pi deployment
├── results/                 # Auto-generated JSONs and figures
├── requirements.txt         
└── README.md
```

---

## 🗺️ Roadmap / Future Work
- [x] Phase 1: 2D Matrix Simulation & Algorithm Decoupling
- [x] Phase 1.5: Real-world Edge Hardware Deployment (Orange Pi)
- [ ] Phase 2: Integration with ROS 2 + Gazebo
- [ ] Phase 2.5: Physical Robot Deployment (e.g., TurtleBot4 / Ackermann platforms) with actual LiDAR.
```