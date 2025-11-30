import pandas as pd
import os

# 项目根路径
PROJECT_ROOT = r"C:\Users\张殊赫\Desktop\TITS-v4.0"
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

# 确保 plots 文件夹存在
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------
# 1. 生成 Sensitivity 表格
# -----------------------------
sensitivity_csv = os.path.join(LOGS_DIR, "sensitivity_results.csv")
sensitivity_tex = os.path.join(PLOTS_DIR, "sensitivity_table.tex")

if os.path.exists(sensitivity_csv):
    df = pd.read_csv(sensitivity_csv)
    df.to_latex(sensitivity_tex, index=False, float_format="%.2f")
    print(f"Sensitivity LaTeX table saved: {sensitivity_tex}")
else:
    print(f"{sensitivity_csv} 不存在，请先生成 CSV 文件。")

# -----------------------------
# 2. 生成 Ablation 表格
# -----------------------------
ablation_csv = os.path.join(LOGS_DIR, "ablation_results.csv")
ablation_tex = os.path.join(PLOTS_DIR, "ablation_table.tex")

if os.path.exists(ablation_csv):
    df = pd.read_csv(ablation_csv)
    df.to_latex(ablation_tex, index=False, float_format="%.2f")
    print(f"Ablation LaTeX table saved: {ablation_tex}")
else:
    print(f"{ablation_csv} 不存在，请先生成 CSV 文件。")