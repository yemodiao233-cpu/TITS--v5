#!/usr/bin/env python3
"""
run_plot.py - 自动化超参数敏感度分析与消融实验脚本

本次修改目标:
1. 严格精确恢复上一版代码的参数扫描范围。
2. 应用全面的 IEEE 期刊美化风格。
3. 优化绘图，解决文字（图例/曲线标签）重叠问题。
4. **新增：测量单时隙决策时延 (Decision Time per Slot)。**
5. **优化：改进端到端时延的收集逻辑。**
6. **优化：删除 run_sensitivity_analysis 中的重复代码块。**
"""

import os
import time
import importlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d  # 引入高斯核卷积平滑
from matplotlib.lines import Line2D  # 用于创建自定义图例行

# 导入 main.py 中的工具函数
# 注意：main.py 需要包含 load_solver, SLOTS 和 timed_solve 函数
import main

# -----------------------------
# 配置
# -----------------------------
TARGET_SOLVER = "solvers.OLMA_Solver_perfect.OLMA_Solver"
PLOT_DIR = "plots/sensitivity"
LOG_DIR = "logs"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 定义重复实验次数
N_RUNS = 20

# 高斯平滑强度
GAUSSIAN_SMOOTHING_SIGMA = 2

# 默认基础配置 (保持不变)
BASE_ENV_CFG = {
    "F_max": 2.0, "B_max": 20.0, "P_max": 1.0,
    "weights": [0.5, 0.3, 0.2]
}
BASE_SOLVER_CFG = {
    "V": 50.0, "I_bcd": 10, "I_sca": 5, "epsilon": 1e-3,
    "ablation": "none"
}


# -----------------------------
# Matplotlib IEEE 美化配置
# -----------------------------
def set_ieee_plot_style():
   """设置符合 IEEE 期刊要求的 Matplotlib 默认参数。"""
   plt.style.use('default')
   plt.rcParams.update({
       'font.size': 12,
       'axes.labelsize': 14,
       'axes.titlesize': 16,
       'xtick.labelsize': 9,  # 缩到最小常用值 9pt
       'ytick.labelsize': 9,  # 缩到最小常用值 9pt
       'legend.fontsize': 9,  # 缩到最小常用值 9pt
       'font.family': 'serif',
       'font.serif': ['Times New Roman', 'serif'],
       'mathtext.fontset': 'cm',
       'axes.linewidth': 1.0,
       'lines.linewidth': 2.0,
       'lines.markersize': 6,
       'grid.linestyle': '--',
       'grid.linewidth': 0.5,
       'grid.alpha': 0.7,
       'savefig.dpi': 600,
       'savefig.format': 'pdf',
       'figure.autolayout': True,
       'axes.unicode_minus': False,
   })
   try:
       plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
   except Exception:
       pass



# -----------------------------
# 核心运行函数 (已修改)
# -----------------------------
def run_single_experiment(env_config_update, solver_config_update, run_name):
    """
    运行单次实验 (一个完整的 SLOTS 周期) 并返回关键指标 summary
    - 测量并返回 Decision_Delay_mean (单时隙决策时延)。
    - 改进 Delay_mean 的收集逻辑，优先使用环境提供的 E2E_Delay。
    """
    # 1. 合并配置
    env_cfg = BASE_ENV_CFG.copy()
    env_cfg.update(env_config_update)
    solver_cfg = BASE_SOLVER_CFG.copy()
    solver_cfg.update(solver_config_update)

    # 2. 初始化环境
    try:
        env_module = importlib.import_module("solvers.environment")
        VEC_Environment = getattr(env_module, "VEC_Environment")
        env = VEC_Environment(env_cfg)
        env.reset()
    except Exception as e:
        print(f"Environment initialization error: {e}")
        return None

    # 3. 初始化求解器
    try:
        solver_obj = main.load_solver(TARGET_SOLVER, env_cfg, solver_cfg)
    except Exception as e:
        print(f"Solver initialization error: {e}")
        return None

    # 4. 运行循环
    metrics = {
        "costs": [], "delays": [], "energies": [],
        "decision_times": []  # 新增：决策时延
    }

    for t in range(main.SLOTS):
        state = env.get_state()

        # === 测量单时隙决策时延 ===
        start_time = time.time()
        decision = main.timed_solve(solver_obj, state)
        end_time = time.time()
        metrics["decision_times"].append(end_time - start_time)
        # =========================

        diag = env.step(decision, state)
        metrics["costs"].append(diag.get("total_cost", 0.0))

        # === 优化端到端时延收集 ===
        # 尝试获取环境返回的严格 E2E_Delay (如果环境支持)
        e2e_delay_strict = diag.get("E2E_Delay", 0.0)

        if e2e_delay_strict > 0.0:
            # 使用环境返回的严格 E2E 时延（假设它记录了已完成任务的平均时延）
            d_total = e2e_delay_strict
        else:
            # 回退到原始的惩罚项时延计算 (队列+传输+处理+回传)
            d_total = diag.get("delay_queue", 0) + diag.get("delay_tx", 0) + \
                      diag.get("delay_proc", 0) + diag.get("delay_backhaul", 0)

        metrics["delays"].append(d_total)
        # =========================

        e_total = diag.get("energy_tx", 0) + diag.get("energy_srv", 0)
        metrics["energies"].append(e_total)

    # 5. 计算统计值
    # 默认将 Delay_mean 视为 E2E_Delay_mean (无论是严格 E2E 还是惩罚项时延)
    summary = {
        "C_mean": np.mean(metrics["costs"]),
        "Delay_mean": np.mean(metrics["delays"]),
        "Energy_mean": np.mean(metrics["energies"]),
        "Decision_Delay_mean": np.mean(metrics["decision_times"]),  # 新增返回
        "E2E_Delay_mean": np.mean(metrics["delays"])  # 适配 ablation_study 现有代码
    }
    return summary

# -----------------------------
# 模块 1: 敏感度分析 (删除重复代码块)
# -----------------------------
def run_sensitivity_analysis():
   print("\n" + "=" * 50)
   print(f">>> 开始超参数敏感度分析 (N_RUNS={N_RUNS})")
   print(">>> 扫描范围已精确恢复到上一版配置 (扩大五倍以上)。")
   print(f">>> 已启用高斯平滑 (sigma={GAUSSIAN_SMOOTHING_SIGMA})。")
   print(">>> 图像将按 IEEE 期刊标准美化。")
   print("=" * 50)

   # 定义要扫描的参数及其范围 (精确恢复上一版配置)
   # (Title, Type, Key, Values, Label, Total_Range)
   experiments = [
       # 1. V: 100 到 1200
       ("Lyapunov Parameter $V$", "solver", "V",
        np.linspace(100, 1200, 21).round(1).tolist(),
        "Lyapunov Parameter ($V$)", 1100),

       # 2. Pmax: 0.1 到 6.0
       ("Pmax (Power Limit)", "env", "P_max",
        np.linspace(0.1, 6.0, 21).round(3).tolist(),
        "Power Limit (W)", 5.9),

       # 3. Fmax (CPU Freq): 1.0 到 11.0
       ("Fmax (CPU Freq)", "env", "F_max",
        np.linspace(1.0, 11.0, 21).round(3).tolist(),
        "Max Frequency (GHz)", 10.0),

       # 4. I_bcd (Iterations): 1 到 80, 步长4
       ("I_bcd (Iterations)", "solver", "I_bcd",
        list(range(1, 81, 4)),
        "BCD Iterations", 79),

       # 5. Epsilon (Convergence): 10^-6 到 10^0, 21点对数刻度
       ("Epsilon (Convergence)", "solver", "epsilon",
        np.logspace(-6, 0, 21).round(9).tolist(),
        "Epsilon (Convergence Threshold)", 1.0),

       # 6. Weight_Energy: 0.0 到 1.0
       ("Weight_Energy", "env", "weights_E",
        np.linspace(0.0, 1.0, 21).round(3).tolist(),
        "Energy Weight ($W_E$)", 1.0)
   ]

   optimal_x_values = {}

   for title, cfg_type, key, values, xlabel, total_range in experiments:
       all_means = []

       print(f"\n--- Testing Sensitivity: {title} ({len(values)} points) ---")

       for v in values:
           env_upd = {}
           slv_upd = {}
           if key == "weights_E":
               remain = 1.0 - v
               # 假设权重分配给 Cost (w_c) 和 Penalty (w_h)
               w_c = remain / 2
               w_h = remain / 2
               env_upd["weights"] = [v, w_c, w_h]
           else:
               if cfg_type == "env":
                   env_upd[key] = v
               else:
                   slv_upd[key] = v

           run_costs = []
           for run in range(N_RUNS):
               summary = run_single_experiment(env_upd, slv_upd, f"{key}={v}_run{run}")
               if summary:
                   run_costs.append(summary["C_mean"])
               else:
                   run_costs.append(np.nan)

           if run_costs:
               all_means.append(run_costs)
           else:
               all_means.append([np.nan] * N_RUNS)

       cost_array = np.array(all_means)
       mean_costs = np.nanmean(cost_array, axis=1)
       std_costs = np.nanstd(cost_array, axis=1)

       # === 核心：寻找原始数据全局最低点并计算平滑曲线 (仅保留一次) ===
       valid_indices = np.where(~np.isnan(mean_costs))[0]
       X_data = np.array(values)[valid_indices]
       Y_data_raw = np.array(mean_costs)[valid_indices]

       smoothed_costs = np.full_like(mean_costs, np.nan)
       opt_x_raw, min_cost_raw = np.nan, np.nan

       # 1. 寻找原始数据的全局最低点
       if len(Y_data_raw) > 0:
           min_idx_raw = np.argmin(Y_data_raw)
           opt_x_raw = X_data[min_idx_raw]  # 原始数据最低点的 X 值
           min_cost_raw = Y_data_raw[min_idx_raw]  # 原始数据最低点的 Y 值
           optimal_x_values[key] = opt_x_raw  # 更新汇总表为原始最低点
           print(f"  -> Raw Global Minimum X: {opt_x_raw:.4g}")
       else:
           print(f"  -> Not enough data points ({len(Y_data_raw)}). Skipping optimum search.")

       # 2. 计算高斯平滑曲线（作为辅助线）
       if len(Y_data_raw) > 1:
           smoothed_costs[valid_indices] = gaussian_filter1d(Y_data_raw, sigma=GAUSSIAN_SMOOTHING_SIGMA)
       # ======================================================================

       # === 保存敏感度扫描数据 ===
       df = pd.DataFrame({
           key: values,
           "C_mean": mean_costs,
           "C_std": std_costs,
           "C_smoothed": smoothed_costs
       })
       filename_prefix = f"sensitivity_{key}"
       csv_path = os.path.join(PLOT_DIR, f"{filename_prefix}.csv")
       df.to_csv(csv_path, index=False)

       # === 绘图 (IEEE 风格) ===
       plt.figure(figsize=(7, 5))

       # 1. 初始化图例句柄和颜色常量（**确保它们首先被定义**）
       RAW_COLOR = '#1E88E5'
       SMOOTH_COLOR = '#D81B60'
       HIGHLIGHT_COLOR = '#4CAF50'

       # 2. 绘制原始数据
       raw_line, = plt.plot(values, mean_costs, marker='o', linestyle='-', markersize=6,
                            color=RAW_COLOR, linewidth=2.0, label='Mean Raw Cost')

       # 3. 初始化 legend_handles 和 legend_labels
       legend_handles = [raw_line,
                         plt.Rectangle((0, 0), 1, 1, fc=RAW_COLOR, alpha=0.15)]
       legend_labels = ['Mean Raw Cost', r'Mean $\pm$ STD']

       # 4. 绘制误差带
       fill_area = plt.fill_between(
           values, mean_costs - std_costs, mean_costs + std_costs,
           color=RAW_COLOR, alpha=0.15
       )

       # 5. 绘制高斯平滑曲线 (并更新图例)
       if len(Y_data_raw) > 1:
           smoothed_line, = plt.plot(values, smoothed_costs, color=SMOOTH_COLOR, linestyle='-.', linewidth=1.5,
                                     label='Gaussian Smoothed Trend')
           legend_handles.append(smoothed_line)
           legend_labels.append('Smoothed Trend (Aux.)')

       # 6. 突出显示原始数据的全局最低点 (并更新图例)
       if not np.isnan(opt_x_raw):
           plt.axvline(opt_x_raw, color=HIGHLIGHT_COLOR, linestyle=':', linewidth=1.5)
           opt_scatter = plt.scatter(opt_x_raw, min_cost_raw, color=HIGHLIGHT_COLOR, marker='X', s=250, zorder=10,
                                     edgecolors='black', linewidths=1.0)
           legend_handles.append(opt_scatter)
           legend_labels.append(f'Global Minimum (X={opt_x_raw:.3g})')

       # 7. 添加 N_Points 和 N_Runs 信息（使用空白句柄）
       # ... (Line2D code for annotations)

       # =======================================================
       # !!! 强制设置 Y 轴范围：0.4 到 1.2 (最终确定位置) !!!
       plt.ylim(0.4, 1.2)

       # 8. 配置图表和图例
       plt.title(f"Sensitivity Analysis: {title}")
       plt.xlabel(xlabel)
       plt.ylabel("Average System Cost")
       plt.grid(True)

       # 集中处理图例
       # 将图例移至图表内部的右下角
       plt.legend(handles=legend_handles, labels=legend_labels,
                  loc='lower right', bbox_to_anchor=(1.0, 0.0), frameon=True,
                  ncol=2)

       # 移除/注释掉 plt.subplots_adjust(top=0.8)
       # 因为图例现在在右侧，不再挤占顶部的空间
       # plt.subplots_adjust(top=0.8)

       # 移除右侧和上侧边框，设置刻度线向内
       plt.gca().spines['right'].set_visible(False)
       plt.gca().spines['top'].set_visible(False)
       plt.tick_params(direction='in')

       # 处理 Epsilon 的对数刻度
       if key == "epsilon":
           plt.xscale('log')

       # 保存为 PDF 和 PNG (增加 pad_inches=0.1 确保捕获图例)
       filename_base = f"sensitivity_{key}"
       plt.savefig(os.path.join(PLOT_DIR, f"{filename_base}.pdf"), bbox_inches='tight', pad_inches=0.1)
       plt.savefig(os.path.join(PLOT_DIR, f"{filename_base}.png"), dpi=600, bbox_inches='tight', pad_inches=0.1)
       plt.close()

   # === 4. 汇总并打印最优值 (供用户参考) ===
   print("\n" + "=" * 60)
   print(">>> 🎯 最优参数值汇总 (基于原始数据的最低点)")
   print("-" * 60)

   results_table = []
   for title, cfg_type, key, values, xlabel, total_range in experiments:
       opt_x = optimal_x_values.get(key, "N/A")

       results_table.append({
           "Parameter": key,
           "Optimal X": f"{opt_x:.4g}" if isinstance(opt_x, (float, int)) else opt_x,
           "Current Range": f"[{values[0]} to {values[-1]}]",
           "N_Points": len(values)
       })

   print(pd.DataFrame(results_table).to_string(index=False))
   print("=" * 60)


# -----------------------------
# 模块 2: 消融实验 (已适配新指标)
# -----------------------------
def run_ablation_study():
    """
    执行消融实验 (Ablation Study)，测试 OLMA 算法各个组件的贡献。
    现在包含 Decision_Delay_mean。
    """
    print("\n" + "=" * 50)
    print(f">>> 开始消融实验 (Ablation Study, N_RUNS={N_RUNS})")
    print(">>> 已启用决策时延 (Decision_Delay_mean) 收集。")
    print("=" * 50)

    # 1. 定义实验变体 (Ablation Variants)
    variants = [
        ("OLMA (Full)", "none"),
        ("w/o Power Control", "no_power"),
        ("w/o Bandwidth Alloc", "no_bw"),
        ("w/o Computation Offloading", "no_offload"),
        ("w/o Freq Scaling", "no_freq")
    ]

    records = []

    for label, mode in variants:
        print(f"Running Ablation: {label} ...")

        slv_upd = {"ablation": mode}

        # 原始指标收集列表
        all_costs = []
        all_delays = []
        all_energies = []

        # 新增指标收集列表
        all_e2e_delays = []
        all_decision_delays = []  # 新增

        for run in range(N_RUNS):
            # 执行单次实验
            summary = run_single_experiment({}, slv_upd, f"{label}_run{run}")

            if summary:
                # 收集原始指标
                all_costs.append(summary["C_mean"])
                all_delays.append(summary["Delay_mean"])
                all_energies.append(summary["Energy_mean"])

                # 收集新增的时延指标
                all_e2e_delays.append(summary.get("E2E_Delay_mean", 0))
                all_decision_delays.append(summary.get("Decision_Delay_mean", 0))  # 收集新指标
            else:
                print(f"  -> WARNING: {label}_run{run} 运行失败，结果为空。")

        if all_costs:
            # 2. 计算 Mean 和 STD，并记录结果
            rec = {
                "Method": label,

                # 原始指标
                "Cost_Mean": np.mean(all_costs),
                "Cost_STD": np.std(all_costs),
                "Delay_Mean": np.mean(all_delays),
                "Delay_STD": np.std(all_delays),
                "Energy_Mean": np.mean(all_energies),
                "Energy_STD": np.std(all_energies),

                # 新增时延指标 (E2E_Delay_Mean 依赖于 run_single_experiment 中的 Delay_mean)
                "E2E_Delay_Mean": np.mean(all_e2e_delays),
                "E2E_Delay_STD": np.std(all_e2e_delays),
                "Decision_Delay_Mean": np.mean(all_decision_delays),  # 新增
                "Decision_Delay_STD": np.std(all_decision_delays),  # 新增
            }
            records.append(rec)
        else:
            print(f"  -> WARNING: {label} 运行失败，跳过。")

    # 3. 保存 CSV 并打印结果
    if records:
        df = pd.DataFrame(records)
        csv_path = os.path.join(LOG_DIR, "ablation_results.csv")
        df.to_csv(csv_path, index=False)
        print("\n" + "-" * 50)
        print(f"消融实验结果已保存至: {csv_path}")
        print("-" * 50)

        # 打印包含所有关键指标的摘要表
        print("📊 关键指标摘要 (Mean Values):")
        print(df[[
            "Method",
            "Cost_Mean",
            "Delay_Mean",
            "E2E_Delay_Mean",
            "Decision_Delay_Mean",  # 打印新指标
            "Energy_Mean"
        ]].to_string())

        print("\n（完整结果包含标准差，已保存至 CSV 文件）")
    else:
        print("消融实验未产生数据。")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # 在程序启动时设置 IEEE 绘图风格
    set_ieee_plot_style()

    print(f">>> Loading environment and solver module from main.py context...")
    print(f">>> Experiments will be run {N_RUNS} times for statistical significance.")

    # 1. 运行敏感度分析
    run_sensitivity_analysis()

    # 2. 运行消融实验
    run_ablation_study()

    print("\n>>> All analysis tasks completed.")

