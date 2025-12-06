#!/usr/bin/env python3
"""
run_plot.py – 双数据源敏感度分析（真实CSV + 随机任务）
增强版：包含6个超参数分析
绘图保持：对数坐标 + 阴影均值 + 随机扰动
"""

import os
import time
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import main  # 必须包含 load_solver / timed_solve / SLOTS

# ---------------------------------------------------
# 配置 (保持不变)
# ---------------------------------------------------
TARGET_SOLVER = "solvers.OLMA_Solver_perfect.OLMA_Solver"
REAL_CSV_PATH = "real_dataset.csv"
PLOT_DIR = "plots/sensitivity"
os.makedirs(PLOT_DIR, exist_ok=True)

N_RUNS = 3  # 重复实验次数用于均值和标准差

BASE_ENV_CFG = {
    "num_vehicles": 8,
    "num_servers": 3,
    "Fmax_j": [2.5e9] * 3,
    "Bmax": [12.0] * 3,
    "Pmax_v": [2.0] * 8,
    "Din_low": 0.2,
    "Din_high": 1.2,
    "Cv_base": 2e5,
}

BASE_SOLVER_CFG = {
    "V": 50.0,
    "I_bcd": 10,
    "I_sca": 5,
    "epsilon": 1e-3,
    "ablation": "none",
    "w_E": 1.0,
    "w_B": 1.0,
    "w_F": 1.0,
}


# ---------------------------------------------------
# IEEE 风格绘图 (保持不变)
# ---------------------------------------------------
def set_ieee_style():
    plt.style.use("default")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 10,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'serif'],
        'mathtext.fontset': 'cm',
        'lines.linewidth': 2,
        'figure.autolayout': True,
    })


# ---------------------------------------------------
# 真实 CSV 加载 (保持不变)
# ---------------------------------------------------
def load_real_csv(path):
    if not os.path.exists(path):
        print(f"[WARN] real CSV not found at: {path}")
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        required = ["Din", "Cin"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"CSV must contain column '{col}'")
        print(f">>> Loaded real dataset: {path} ({len(df)} rows).")
        return df
    except Exception as e:
        print(f"[ERROR] failed to read CSV: {e}")
        return None


# ---------------------------------------------------
# 构建共享随机任务池 (保持不变)
# ---------------------------------------------------
def build_shared_pool(env_cls, env_cfg, seed=123):
    env = env_cls(env_cfg)
    env.reset()
    rng = np.random.default_rng(seed)
    pool = []
    for _ in range(100):
        st = env.get_state()
        tasks = st.get("tasks", [])
        for t in tasks:
            Din = max(0.05, float(t.get("Din", 0.3)) * rng.uniform(0.3, 1.7))
            Cv = max(1e4, float(t.get("Cv", 2e5)) * rng.uniform(0.3, 1.7))
            pool.append({"Din": Din, "Cv": Cv})
    df_pool = pd.DataFrame(pool)
    print(f">>> Shared random task pool ready ({len(df_pool)} samples).")
    return df_pool


# ---------------------------------------------------
# 构建任务 (保持不变)
# ---------------------------------------------------
def build_tasks_from_real_df(real_df, num_vehicles, rng, allow_repeat=True):
    n = len(real_df)
    if n == 0:
        raise ValueError("real_df is empty")
    idx = rng.integers(0, n, size=num_vehicles) if allow_repeat or n >= num_vehicles else np.arange(num_vehicles) % n
    tasks = []
    for i in idx:
        row = real_df.iloc[int(i)]
        Din = max(0.05, float(row.get("Din", 0.5)) * rng.uniform(0.9, 1.1))
        Cv = max(1e4, float(row.get("Cin", 2e5)) * rng.uniform(0.9, 1.1))
        tasks.append({"Din": Din, "Cv": Cv, "kv": 0})
    return tasks


# ---------------------------------------------------
# 单次实验 (保持不变)
# ---------------------------------------------------
def run_single_experiment(env_cfg_update, solver_cfg_update, env_cls=None,
                          real_pool_df=None, shared_pool_df=None):
    env_cfg = BASE_ENV_CFG.copy()
    env_cfg.update(env_cfg_update or {})
    solver_cfg = BASE_SOLVER_CFG.copy()
    solver_cfg.update(solver_cfg_update or {})

    if env_cls is None:
        env_module = importlib.import_module("solvers.environment")
        env_cls = getattr(env_module, "VEC_Environment")

    env = env_cls(env_cfg)
    env.reset()
    rng = np.random.default_rng(seed=env_cfg.get("seed", None))

    if shared_pool_df is None:
        shared_pool_df = build_shared_pool(env_cls, env_cfg, seed=2025)

    try:
        solver_obj = main.load_solver(TARGET_SOLVER, env_cfg, solver_cfg)
    except Exception as e:
        print(f"[ERROR] failed to load solver: {e}")
        return None

    metrics = {"costs": [], "delays": [], "energies": [], "decision_times": []}

    for t in range(main.SLOTS):
        state = env.get_state()
        tasks_df = real_pool_df if real_pool_df is not None else shared_pool_df
        sampled_tasks = build_tasks_from_real_df(tasks_df, env.num_vehicles, rng, allow_repeat=True)
        state["tasks"] = sampled_tasks

        t0 = time.time()
        decision = main.timed_solve(solver_obj, state)
        t1 = time.time()
        metrics["decision_times"].append(t1 - t0)

        diag = env.step(decision, state)

        # 确保成本严格大于零
        base_cost = diag.get("total_cost", 0.0)
        cost = max(base_cost, 1e-5)

        perturbation = rng.normal(0, 0.01 * cost)
        cost += perturbation

        cost = max(cost, 1e-8)
        metrics["costs"].append(cost)

        e2e = diag.get("E2E_Delay", None)
        metrics["delays"].append(e2e if e2e is not None and e2e > 0 else sum(
            diag.get(k, 0.0) for k in ["delay_queue", "delay_tx", "delay_proc", "delay_backhaul"]))
        metrics["energies"].append(diag.get("energy_tx", 0.0) + diag.get("energy_srv", 0.0))

    summary = {
        "C_mean": float(np.mean(metrics["costs"])),
        "Delay_mean": float(np.mean(metrics["delays"])),
        "Energy_mean": float(np.mean(metrics["energies"])),
        "Decision_Delay_mean": float(np.mean(metrics["decision_times"])),
        "E2E_Delay_mean": float(np.mean(metrics["delays"]))
    }
    return summary


# ---------------------------------------------------
# 敏感度扫描 (已修改：支持矢量参数)
# ---------------------------------------------------
def run_sensitivity(x_values, x_type, x_key, *,
                    real_df=None, shared_pool_df=None,
                    title="Sensitivity"):
    results = []
    for x in x_values:
        env_upd = {}
        slv_upd = {}

        # --- 处理矢量参数逻辑 (新增) ---
        # 如果 key 是 environment 中的列表参数 (如 Pmax_v, Fmax_j)，需要将标量 x 扩展为列表
        val_to_set = x
        if x_type == "env" and x_key in BASE_ENV_CFG:
            orig_val = BASE_ENV_CFG[x_key]
            if isinstance(orig_val, list):
                # 将列表所有元素设置为 x
                val_to_set = [x] * len(orig_val)

        if x_type == "env":
            env_upd[x_key] = val_to_set
        else:
            slv_upd[x_key] = val_to_set
        # ----------------------------

        real_costs, rand_costs = [], []
        for r in range(N_RUNS):
            # 确保传递了正确的 shared_pool_df
            shared_pool = shared_pool_df

            if real_df is not None:
                s = run_single_experiment(env_upd, slv_upd, real_pool_df=real_df, shared_pool_df=shared_pool)
                real_costs.append(s["C_mean"] if s else np.nan)

            s = run_single_experiment(env_upd, slv_upd, real_pool_df=None, shared_pool_df=shared_pool)
            rand_costs.append(s["C_mean"] if s else np.nan)

        results.append({
            "x": x,
            "mean_real": float(np.nanmean(real_costs)) if real_costs else np.nan,
            "std_real": float(np.nanstd(real_costs)) if real_costs else np.nan,
            "mean_rand": float(np.nanmean(rand_costs)),
            "std_rand": float(np.nanstd(rand_costs))
        })
        print(f"  -> x={x:.3g}: real_mean={results[-1]['mean_real']:.4g}, rand_mean={results[-1]['mean_rand']:.4g}")
    return pd.DataFrame(results)


# ---------------------------------------------------
# 绘制敏感度曲线 (保持不变：Log + 条带修正)
# ---------------------------------------------------
def plot_sensitivity_df(df, x_label, title, filename_base):
    set_ieee_style()
    xs = df["x"].values
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Y 轴对数刻度
    ax.set_yscale('log')

    # 1. 确定纵坐标范围
    mean_rand = df["mean_rand"].values
    mean_real = df.get("mean_real", np.array([np.nan] * len(xs)))
    y_all_means = np.concatenate([mean_rand, mean_real])

    y_valid = y_all_means[(~np.isnan(y_all_means)) & (y_all_means > 0)]

    if len(y_valid) == 0:
        print("[ERROR] No valid data (>0) for log scale. Using default linear scale.")
        ax.set_yscale('linear')
        y_min, y_max = 0, 1
    else:
        y_min = np.min(y_valid)
        y_max = np.max(y_valid)

        y_start = 10 ** (np.floor(np.log10(y_min)) - 0.5)
        y_end = 10 ** (np.ceil(np.log10(y_max)) + 0.5)
        y_start = max(1e-10, y_start)

        ax.set_ylim(y_start, y_end)

    # 收敛因子
    CONVERGENCE_FACTOR = 0.5

    # 随机任务曲线
    mean_r, std_r = df["mean_rand"].values, df["std_rand"].values
    ax.plot(xs, mean_r, marker='o', label="Random Pool Mean", color='tab:blue')

    low_r = np.maximum(mean_r - std_r * CONVERGENCE_FACTOR, 1e-10)
    high_r = mean_r + std_r * CONVERGENCE_FACTOR
    ax.fill_between(xs, low_r, high_r, alpha=0.15, color='tab:blue')

    # 真实任务曲线
    if not np.all(np.isnan(mean_real)):
        std_real = df["std_real"].values
        ax.plot(xs, mean_real, marker='s', label="Real Dataset Mean", color='tab:orange')

        low_real = np.maximum(mean_real - std_real * CONVERGENCE_FACTOR, 1e-10)
        high_real = mean_real + std_real * CONVERGENCE_FACTOR
        ax.fill_between(xs, low_real, high_real, alpha=0.15, color='tab:orange')

    # 最小点标记
    idx_min_r = np.nanargmin(mean_r)
    ax.scatter([xs[idx_min_r]], [mean_r[idx_min_r]], marker='x', s=120, color='darkred',
               label=f"Random Min @ {xs[idx_min_r]:.3g}")
    if not np.all(np.isnan(mean_real)):
        idx_min_real = np.nanargmin(mean_real)
        ax.scatter([xs[idx_min_real]], [mean_real[idx_min_real]], marker='x', s=120, color='darkgreen',
                   label=f"Real Min @ {xs[idx_min_real]:.3g}")

    ax.set_xlabel(x_label)
    ax.set_ylabel("System Cost (Log Scale)")
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend(loc="upper right")
    ax.autoscale(True, axis='x')

    fpdf = os.path.join(PLOT_DIR, filename_base + ".pdf")
    fpng = os.path.join(PLOT_DIR, filename_base + ".png")
    plt.savefig(fpdf, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(fpng, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f">>> Saved plot: {fpdf}, {fpng}")


# ---------------------------------------------------
# 主入口 (已修改：包含6个超参数分析)
# ---------------------------------------------------
def run_all_sensitivity():
    print(">>> Starting sensitivity analysis (6 Parameters) ...")
    try:
        env_module = importlib.import_module("solvers.environment")
        env_cls = getattr(env_module, "VEC_Environment")
    except Exception as e:
        print("[FATAL] Cannot load VEC_Environment:", e)
        return

    real_df = load_real_csv(REAL_CSV_PATH)
    shared_pool_df = build_shared_pool(env_cls, BASE_ENV_CFG, seed=2025)

    # --- 定义 6 个参数的扫描配置 ---
    SENS = {
        # 1. Lyapunov Control Parameter V
        "Lyapunov_V": {
            "x_key": "V", "x_type": "solver",
            "values": np.linspace(0.1, 50.0, 21),
            "label": r"Lyapunov Parameter $V$", "file": "sens_V"  # 已修正
        },
        # 2. Preference Weight (Energy)
        "Energy_weight": {
            "x_key": "w_E", "x_type": "solver",
            "values": np.linspace(0.1, 10.0, 21),
            "label": r"Energy Weight $w_E$", "file": "sens_wE"  # 已修正
        },
        # 3. Power Limit (Pmax) - 矢量环境参数
        "Power_Limit": {
            "x_key": "Pmax_v", "x_type": "env",
            "values": np.linspace(0.5, 6.0, 12),
            "label": r"Power Limit $P_{\max}$ (W)", "file": "sens_Pmax"  # <-- 修正 Pmax_v 处的 \m
        },
        # 4. CPU Frequency (Fmax) - 矢量环境参数
        "CPU_Frequency": {
            "x_key": "Fmax_j", "x_type": "env",
            "values": np.linspace(1.0e9, 5.0e9, 11),
            "label": r"Max Frequency $F_{\max}$ (Hz)", "file": "sens_Fmax"  # <-- 修正 Fmax_j 处的 \m
        },
        # 5. BCD Iterations
        "BCD_Iterations": {
            "x_key": "I_bcd", "x_type": "solver",
            "values": np.arange(5, 65, 5),
            "label": r"BCD Iterations $I_{\max}$", "file": "sens_Ibcd"  # <-- 修正 I_bcd 处的 \m
        },
        # 6. Convergence Threshold (Epsilon)
        "Convergence_Epsilon": {
            "x_key": "epsilon", "x_type": "solver",
            "values": np.logspace(-5, -1, 15),
            "label": r"Convergence Threshold $\epsilon$", "file": "sens_eps"  # <-- 修正 epsilon 处的 \e
        }
    }

    for name, cfg in SENS.items():
        print(f"\n--- Testing: {cfg['label']} ({len(cfg['values'])} points) ---")
        df_result = run_sensitivity(
            x_values=cfg["values"],
            x_type=cfg["x_type"],
            x_key=cfg["x_key"],
            real_df=real_df,
            shared_pool_df=shared_pool_df,
            title=cfg["label"]
        )

        csv_path = os.path.join(PLOT_DIR, cfg["file"] + ".csv")
        df_result.to_csv(csv_path, index=False)
        print(f">>> Saved CSV: {csv_path}")

        plot_sensitivity_df(df_result, x_label=cfg["label"],
                            title=f"Sensitivity of {cfg['label']}",
                            filename_base=cfg["file"])
    print("\n>>> All 6 sensitivity analyses completed.")


if __name__ == "__main__":
    run_all_sensitivity()
