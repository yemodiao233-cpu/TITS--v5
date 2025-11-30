# utils/plot_utils.py
import os
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_hyperparam_curve(csv_path, output_dir, param_col, metrics):
    """
    绘制超参数敏感度曲线
    csv_path: logs/sensitivity_results.csv
    output_dir: plots/sensitivity/
    param_col: 超参数列名, e.g., "gamma"
    metrics: dict, key=metric名, value=列名, e.g.,
             {"Avg. Cost": "avg_cost", "Energy/Task": "energy_per_task"}
    """
    ensure_dir(output_dir)
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(6, 4))
    for label, col in metrics.items():
        plt.plot(df[param_col], df[col], marker='o', label=label)
    plt.xlabel(param_col)
    plt.ylabel("Metric Value")
    plt.title(f"{param_col} Sensitivity Analysis")
    plt.grid(True)
    plt.legend()
    save_path = os.path.join(output_dir, f"{param_col}_curve.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved figure: {save_path}")


def plot_ablation_bar(csv_path, output_dir, metric_col, title="Ablation Study"):
    """
    绘制消融实验柱状图
    csv_path: logs/ablation_results.csv
    metric_col: 选择绘制的指标列，如 "avg_cost"
    """
    ensure_dir(output_dir)
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(6, 4))
    plt.bar(df['Method'], df[metric_col], color='skyblue')
    plt.xlabel("Method")
    plt.ylabel(metric_col)
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_path = os.path.join(output_dir, f"{metric_col}_ablation.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved figure: {save_path}")