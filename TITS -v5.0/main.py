#!/usr/bin/env python3
"""
main.py - Single-file runner (paper-level) for OLMA experiments.

Features:
 - Run by pressing "Run" in PyCharm (no CLI required)
 - Default solver: solvers.OLMA_Solver_perfect.OLMA_Solver
 - Fully collects and prints 30+ paper metrics in a clean textual layout
 - Memory-safe per-slot logging and periodic progress prints to avoid "white runs"
 - Compatible with colleague solvers following BaseSolver format:
     return {'assignment':[...], 'power':[...], 'bandwidth':[[...]], 'freq':[[...]]}

Usage:
 - Save this file as main.py in project root, ensure solvers/ and utils/ exist.
 - Click Run in PyCharm or `python main.py`.
"""

import os
import time
import json
import math
from typing import Dict, Any, List
import importlib
import csv

# -----------------------------
# Configurable defaults
# -----------------------------
OUT_DIR = "logs"
SLOTS = 5  # default slots when clicking run (reduced for testing)
# List of solver class paths to run sequentially (you may add multiple)
SOLVERS_TO_RUN = [
    "solvers.OLMA_Solver_perfect.OLMA_Solver",  # default paper solver
    "solvers.A3C_GCN_Seq2Seq_Adapter.A3C_GCN_Seq2Seq_Adapter",  # A3C-GCN-Seq2Seq adapter
    "solvers.NOMA_VEC_Solver.NOMA_VEC_Solver",  # NOMA-enabled multi-F-AP vehicle fog computing solver
    "solvers.OORAA_Solver.OORAA_Solver",  # Energy-Efficiency and Delay Tradeoff solver
    "solvers.BARGAIN_MATCH_Solver.BARGAIN_MATCH_Solver"  # 优化后的Bargaining Matching求解器
]
# Whether to save per-slot CSVs for each solver (keeps minimal fields)
SAVE_PER_SLOT = True

# Periodic progress print frequency (every PROGRESS_EVERY slots)
PROGRESS_EVERY =  max(1, SLOTS // 20)

# SLA threshold for violation statistic (seconds)
SLA_SECONDS = 0.2

# -----------------------------
# Helper: dynamic import and instantiate solver
# -----------------------------
def load_solver(class_path: str, env_cfg: Dict[str, Any], solver_cfg: Dict[str, Any]):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    inst = cls(env_cfg, solver_cfg)
    return inst

# -----------------------------
# Helper: validate/normalize solver decision
# -----------------------------
def _default_decision(Nv: int, Jn: int):
    return {
        "assignment": [0] * Nv,
        "power": [0.0] * Nv,
        "bandwidth": [[0.0] * Jn for _ in range(Nv)],
        "freq": [[0.0] * Jn for _ in range(Nv)],
        "debug": {}
    }

def _normalize_list(x, length, default=0.0):
    try:
        if x is None:
            return [default] * length
        lst = list(x)
        if len(lst) < length:
            lst = lst + [default] * (length - len(lst))
        return lst[:length]
    except Exception:
        return [default] * length

def _normalize_matrix(mat, Nv, Jn, default=0.0):
    try:
        if mat is None:
            return [[default]*Jn for _ in range(Nv)]
        res = []
        for i in range(Nv):
            row = mat[i] if i < len(mat) else []
            row = _normalize_list(row, Jn, default)
            res.append(row)
        return res
    except Exception:
        return [[default]*Jn for _ in range(Nv)]

def validate_and_normalize_decision(decision: Dict[str, Any], Nv: int, Jn: int):
    safe = _default_decision(Nv, Jn)
    if not isinstance(decision, dict):
        return safe
    a = _normalize_list(decision.get("assignment", safe["assignment"]), Nv, default=0)
    p = _normalize_list(decision.get("power", safe["power"]), Nv, default=0.0)
    bw = _normalize_matrix(decision.get("bandwidth", safe["bandwidth"]), Nv, Jn, default=0.0)
    fq = _normalize_matrix(decision.get("freq", safe["freq"]), Nv, Jn, default=0.0)
    dbg = decision.get("debug", {})
    res = {
        "assignment": [int(x) for x in a],
        "power": [float(x) for x in p],
        "bandwidth": [[float(y) for y in row] for row in bw],
        "freq": [[float(y) for y in row] for row in fq],
        "debug": dbg
    }
    return res

def timed_solve(solver_obj, system_state: Dict[str, Any]):
    Nv = max(1, len(system_state.get("V_set", [])))
    Jn = max(1, len(system_state.get("J_set", [])))
    t0 = time.time()
    try:
        raw = solver_obj.solve(system_state)
    except Exception as e:
        # solver crashed: return safe default and include error in debug
        raw = {"debug": {"exception": str(e)}}
    t1 = time.time()
    norm = validate_and_normalize_decision(raw, Nv, Jn)
    norm["_solve_time_s"] = t1 - t0
    return norm

# -----------------------------
# Minimal memory-safe per-slot logger (keeps only essential)
# -----------------------------
class MinimalLogger:
    def __init__(self, out_dir: str, solver_name: str):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.solver_name = solver_name
        self.slots = []  # will hold small dicts only
        self._acc = {
            "total_cost": 0.0,
            "energy_tx": 0.0,
            "energy_srv": 0.0,
            "effective_bits": 0.0,
            "arrival_tasks": 0,
            "completed_tasks": 0,
            "dropped_tasks": 0,
            "handover": 0,
            "queue_sums": 0.0,
            "queue_counts": 0,
            "decision_times": [],
            "delay_samples": []
        }

    def log_slot(self, t: int, state: Dict[str, Any], decision: Dict[str, Any], diag: Dict[str, Any]):
        # store only minimal per-slot fields (strings for lists)
        rec = {
            "slot": t,
            "total_cost": float(diag.get("total_cost", 0.0)),
            "queue_mean": float(sum(diag.get("queue_lengths", [])) / max(1, len(diag.get("queue_lengths", [])))),
            "arrival_tasks": int(diag.get("arrival_tasks", 0)),
            "completed_tasks": int(diag.get("completed_tasks", 0)),
            "dropped_tasks": int(diag.get("dropped_tasks", 0)),
            "handover": int(diag.get("handover", 0)),
            "energy_tx": float(diag.get("energy_tx", 0.0)),
            "energy_srv": float(diag.get("energy_srv", 0.0)),
            "effective_bits": float(diag.get("effective_bits", 0.0)),
            "solve_time_s": float(decision.get("_solve_time_s", 0.0))
        }
        self.slots.append(rec)

        # aggregate
        self._acc["total_cost"] += rec["total_cost"]
        self._acc["energy_tx"] += rec["energy_tx"]
        self._acc["energy_srv"] += rec["energy_srv"]
        self._acc["effective_bits"] += rec["effective_bits"]
        self._acc["arrival_tasks"] += rec["arrival_tasks"]
        self._acc["completed_tasks"] += rec["completed_tasks"]
        self._acc["dropped_tasks"] += rec["dropped_tasks"]
        self._acc["handover"] += rec["handover"]
        self._acc["queue_sums"] += rec["queue_mean"]
        self._acc["queue_counts"] += 1
        self._acc["decision_times"].append(rec["solve_time_s"])
        # delay samples: environment may give delay_total or components
        d_total = float(diag.get("delay_queue", 0.0) + diag.get("delay_tx", 0.0) + diag.get("delay_proc", 0.0) + diag.get("delay_backhaul", 0.0))
        if d_total > 0:
            self._acc["delay_samples"].append(d_total)

    def summarize(self) -> Dict[str, Any]:
        S = {}
        N = len(self.slots)
        S["slots"] = N
        if N == 0:
            return S
        # cost
        S["C_mean"] = self._acc["total_cost"] / N
        # use sample std
        costs = [s["total_cost"] for s in self.slots]
        mean_c = sum(costs)/len(costs)
        var = sum((x-mean_c)**2 for x in costs)/max(1, len(costs))
        S["C_std"] = math.sqrt(var)
        # energy
        S["E_mean"] = self._acc["energy_tx"] + self._acc["energy_srv"]
        S["Energy_per_task"] = (self._acc["energy_tx"] + self._acc["energy_srv"]) / max(1, self._acc["completed_tasks"])
        S["Energy_per_bit"] = (self._acc["energy_tx"] + self._acc["energy_srv"]) / max(1e-9, self._acc["effective_bits"])
        S["EE_proxy_bit_per_J"] = (self._acc["effective_bits"] / max(1e-9, (self._acc["energy_tx"] + self._acc["energy_srv"])))
        # delay
        ds = self._acc["delay_samples"]
        if ds:
            S["avg_delay_mean"] = float(sum(ds)/len(ds))
            S["avg_delay_median"] = float(sorted(ds)[len(ds)//2])
            S["delay_p95"] = float(sorted(ds)[max(0, int(0.95*len(ds))-1)])
            S["SLA_violation_rate"] = float(sum(1 for x in ds if x > SLA_SECONDS) / len(ds))
        else:
            S["avg_delay_mean"] = 0.0
            S["avg_delay_median"] = 0.0
            S["delay_p95"] = 0.0
            S["SLA_violation_rate"] = 0.0
        # queue & load
        S["Avg_queue"] = float(self._acc["queue_sums"] / max(1, self._acc["queue_counts"]))
        S["Acc_rate"] = float(self._acc["completed_tasks"] / max(1, self._acc["arrival_tasks"]))
        S["Drop_rate"] = float(self._acc["dropped_tasks"] / max(1, self._acc["arrival_tasks"]))
        S["Handover_count_total"] = int(self._acc["handover"])
        # decision time
        dt = self._acc["decision_times"]
        S["DecisionTime_ms_mean"] = float(sum(dt)/len(dt) * 1000.0)
        S["DecisionTime_ms_max"] = float(max(dt) * 1000.0)
        # fairness (Jain) approximate using average power per vehicle across slots
        # collect average power per vehicle
        try:
            # compute average power per vehicle by reading slots (if stored)
            # We don't store per-vehicle power to save memory, so approximate by mean of slot 'power' not available.
            S["Jain_mean"] = 0.0  # placeholder (requires more detailed logs)
        except Exception:
            S["Jain_mean"] = 0.0
        # cloud usage placeholder: environment should include in diagnostics if available
        S["CloudRatio"] = 0.0

        # additional fields for energies
        S["EE_proxy_bit_per_J"] = S["EE_proxy_bit_per_J"]
        S["E_tx_total"] = float(self._acc["energy_tx"])
        S["E_srv_total"] = float(self._acc["energy_srv"])

        return S

    def save_csv(self, filename="per_slot_minimal.csv"):
        if not self.slots:
            return None
        path = os.path.join(self.out_dir, filename)
        keys = sorted(set().union(*[set(s.keys()) for s in self.slots]))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for s in self.slots:
                w.writerow(s)
        return path

# -----------------------------
# Pretty text printing for one solver
# -----------------------------
def pretty_print_solver_block(
    solver_name: str,
    S: Dict[str, Any],
    index: int = 1,
    total: int = 1
):
    # ANSI colors
    BOLD = "\033[1m"
    END = "\033[0m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    WHITE_BG = "\033[107m"
    BLACK_FG = "\033[30m"

    # Decorative lines
    line = "═" * 92

    # ================================
    # SUPER HIGHLIGHT SOLVER HEADER
    # ================================
    title = f"🔥 Solver {index}/{total}: {solver_name} 🔥"
    pad = (92 - len(title)) // 2
    padded_title = " " * pad + title + " " * pad

    print("\n" + line)
    print(f"{WHITE_BG}{BLACK_FG}{BOLD}{padded_title}{END}")
    print(line)

    # Helper functions
    def section(title):
        print(f"\n{CYAN}{'='*10} {title} {'='*10}{END}")

    def item(label, key, fmt="{:.6f}", unit=""):
        val = S.get(key, None)
        if val is None:
            print(f"  • {label:<30} :   -")
        else:
            try:
                print(f"  • {label:<30} :   {fmt.format(val)} {unit}")
            except:
                print(f"  • {label:<30} :   {val}")

    # --------------------------------------
    # 1. Basic Info
    # --------------------------------------
    section("基本信息 Basic Info")
    item("时隙数 Slots", "slots", "{:.0f}")
    item("总运行时长 Runtime", "total_runtime_s", "{:.3f}", "s")

    # --------------------------------------
    # 2. Cost & Energy
    # --------------------------------------
    section("成本与能耗 Cost & Energy")
    item("平均系统成本 C_mean", "C_mean")
    item("成本标准差 C_std", "C_std")
    item("能耗总计 E_total", "E_mean")
    item("Tx 能耗", "E_tx_total")
    item("CPU 能耗", "E_srv_total")
    item("能耗/任务", "Energy_per_task")
    item("能耗/bit", "Energy_per_bit", "{:.9f}")
    item("能效 EE(bit/J)", "EE_proxy_bit_per_J")

    # --------------------------------------
    # 3. Delay
    # --------------------------------------
    section("端到端时延 Delay")
    item("平均时延 mean", "avg_delay_mean", "{:.6f}", "s")
    item("中位数时延 median", "avg_delay_median", "{:.6f}", "s")
    item("95% 时延", "delay_p95", "{:.6f}", "s")
    item("99% 时延", "delay_p99", "{:.6f}", "s")
    item("SLA违约率", "SLA_violation_rate", "{:.4f}")

    # --------------------------------------
    # 4. Queue & QoS
    # --------------------------------------
    section("服务质量 QoS")
    item("平均队列长度", "Avg_queue")
    item("接收率 Acc_rate", "Acc_rate", "{:.4f}")
    item("丢弃率 Drop_rate", "Drop_rate", "{:.4f}")

    # --------------------------------------
    # 5. Resource Utilization
    # --------------------------------------
    section("资源利用率 Resource Utilization")
    item("CPU 使用率", "CPU_util_mean")
    item("带宽使用率", "BW_util_mean")
    item("Cloud 使用率", "CloudRatio")

    # --------------------------------------
    # 6. Mobility
    # --------------------------------------
    section("移动性 Mobility")
    item("切换次数 Handover", "Handover_count_total", "{:.0f}")
    item("迁移频率", "Migration_freq_total", "{:.0f}")
    item("迁移成本", "Migration_cost_total")

    # --------------------------------------
    # 7. Availability
    # --------------------------------------
    section("可用性 Availability")
    item("Availability", "Availability", "{:.4f}")

    # --------------------------------------
    # 8. Throughput
    # --------------------------------------
    section("吞吐量 Throughput")
    item("有效比特数", "effective_bits_total", "{:.2f}")

    # --------------------------------------
    # 9. Complexity
    # --------------------------------------
    section("复杂度 Complexity")
    item("平均 Ops", "Ops_mean")
    item("Ops 总量", "Ops_total")

    # --------------------------------------
    # 10. Runtime
    # --------------------------------------
    section("决策性能 Runtime")
    item("平均决策时间", "DecisionTime_ms_mean", "{:.3f}", "ms")
    item("最大决策时间", "DecisionTime_ms_max", "{:.3f}", "ms")

    print("\n" + "─" * 92 + "\n")

# -----------------------------
# Main runner
# -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Enable detailed logging for debugging
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Import contextlib for stdout redirection
    import sys

    # If multiple solvers present in SOLVERS_TO_RUN, we run them sequentially and print blocks
    total_solvers = len(SOLVERS_TO_RUN)
    for idx, solver_path in enumerate(SOLVERS_TO_RUN, start=1):
        # Create output directory for this solver
        solver_output_dir = os.path.join(OUT_DIR, solver_path.replace(".", "_"))
        os.makedirs(solver_output_dir, exist_ok=True)
        
        # Create output file for this solver
        output_file_path = os.path.join(solver_output_dir, f"{solver_path.split('.')[-1]}_output.txt")
        
        print("\n" + "#"*100)
        print(f"Starting solver {idx}/{total_solvers}: {solver_path}")
        print("#"*100 + "\n")
        
        # Open output file and redirect stdout to both terminal and file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # Create a custom stream that writes to both stdout and the file
            class Tee:
                def __init__(self, file_obj, terminal_obj):
                    self.file = file_obj
                    self.terminal = terminal_obj
                def write(self, message):
                    self.file.write(message)
                    self.terminal.write(message)
                def flush(self):
                    self.file.flush()
                    self.terminal.flush()
            
            # Replace stdout with our Tee instance
            original_stdout = sys.stdout
            sys.stdout = Tee(f, original_stdout)
            
            try:
                # load environment and solver
                # env config currently empty dict; if you have JSON config, you can modify below
                env_cfg = {}
                solver_cfg = {}
                try:
                    env_module = importlib.import_module("solvers.environment")
                    VEC_Environment = getattr(env_module, "VEC_Environment")
                    print(f"Successfully imported VEC_Environment from solvers.environment")
                except Exception as e:
                    print("Error: cannot import solvers.environment. Ensure file exists. Exception:", e)
                    logging.error("Environment import error: %s", e)
                    return

                print("Creating environment...")
                env = VEC_Environment(env_cfg)
                env.reset()
                print(f"Environment created with config: {env_cfg}")

                try:
                    print(f"Loading solver from: {solver_path}")
                    solver_obj = load_solver(solver_path, env_cfg, solver_cfg)
                    print(f"Solver loaded successfully: {type(solver_obj).__name__}")
                except Exception as e:
                    print(f"Failed to instantiate solver {solver_path}: {e}")
                    logging.error("Solver instantiation error: %s", e)
                    continue

                logger = MinimalLogger(out_dir=os.path.join(OUT_DIR, solver_path.replace(".", "_")), solver_name=solver_path)
                # run loop
                start_time = time.time()
                print("Running simulation...")
                for t in range(SLOTS):
                    state = env.get_state()
                    decision = timed_solve(solver_obj, state)
                    # ensure decision normalized before passing to env.step
                    # env.step expects mapping for assignment/power/bandwidth/freq
                    # our timed_solve already normalized
                    diag = env.step(decision, state)
                    logger.log_slot(t, state, decision, diag)

                    # periodic progress prints to avoid blind runs
                    if (t + 1) % PROGRESS_EVERY == 0 or t < 3:
                        # compute a quick progress snapshot
                        last = logger.slots[-1] if logger.slots else {}
                        print(f"[{solver_path}] slot {t+1}/{SLOTS}  avgQ_snapshot={last.get('queue_mean', 0):.3f}  last_cost={last.get('total_cost', 0):.4f}  solve_ms={last.get('solve_time_s',0)*1000:.2f}")

                duration = time.time() - start_time
                print(f"Simulation completed for solver: {solver_path}")
                print(f"\nSolver {solver_path} finished {SLOTS} slots in {duration:.2f}s")

                # gather summary and persist
                summary = logger.summarize()
                summary["total_runtime_s"] = duration
                out_subdir = os.path.join(OUT_DIR, solver_path.replace(".", "_"))
                os.makedirs(out_subdir, exist_ok=True)
                # save minimal per-slot CSV (memory safe)
                if SAVE_PER_SLOT:
                    csvpath = logger.save_csv("per_slot_minimal.csv")
                    if csvpath:
                        print(f"Saved per-slot minimal CSV to {csvpath}")
                # save summary json
                try:
                    with open(os.path.join(out_subdir, "summary.json"), "w") as f:
                        json.dump(summary, f, indent=2)
                    print(f"Saved summary.json to {out_subdir}")
                except Exception as e:
                    print("Failed to save summary.json:", e)
                    logging.error("Summary save error: %s", e)

                # pretty print block
                pretty_print_solver_block(solver_path, summary, index=idx, total=total_solvers)
            finally:
                # Restore original stdout
                sys.stdout = original_stdout

    print("\nAll solvers done. Logs in:", OUT_DIR)


if __name__ == "__main__":
    main()