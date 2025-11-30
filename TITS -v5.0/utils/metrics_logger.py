import numpy as np
import csv
import os
from typing import Dict, Any, List


class MetricsLogger:
    """
    最终版 TITS 论文指标日志器。
    环境 step() 返回的所有字段 + 扩展指标（共 30+）
    将在 summarize() 计算并输出。

    兼容 main.py / solvers / environment.py
    """

    def __init__(self, out_dir: str = "logs"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        # 逐时隙记录
        self.slots: List[Dict[str, Any]] = []

        # 全局累积变量
        self.total_tasks = 0
        self.completed_tasks = 0
        self.dropped_tasks = 0

        # 延迟
        self.delay_samples = []

        # 切换
        self.handover_count = 0

        # 能耗
        self.energy_tx_total = 0.0
        self.energy_srv_total = 0.0

        # 吞吐
        self.effective_bits_total = 0.0

        # resource utilization
        self.cpu_util_samples = []
        self.bw_util_samples = []

        # migration
        self.migration_freq_total = 0
        self.migration_cost_total = 0.0

        # availability
        self.availability_samples = []

        # ops（复杂度）
        self.ops_samples = []

        # timing
        self.solve_times = []

    # ---------------------------------------------------------------------
    # 记录单个时隙
    # ---------------------------------------------------------------------
    def log_slot(self, slot_idx: int,
                 state: Dict[str, Any],
                 decision: Dict[str, Any],
                 diag: Dict[str, Any]):

        rec = {
            "slot": slot_idx,
            "total_cost": diag.get("total_cost", 0.0),
            "queue_len": diag.get("queue_lengths", []),

            # delay
            "delay_queue": diag.get("delay_queue", 0.0),
            "delay_tx": diag.get("delay_tx", 0.0),
            "delay_proc": diag.get("delay_proc", 0.0),

            "energy_tx": diag.get("energy_tx", 0.0),
            "energy_srv": diag.get("energy_srv", 0.0),

            "arrival_tasks": diag.get("arrival_tasks", 0),
            "completed_tasks": diag.get("completed_tasks", 0),
            "dropped_tasks": diag.get("dropped_tasks", 0),

            "handover": diag.get("handover", 0),
            "effective_bits": diag.get("effective_bits", 0.0),

            # new 资源利用
            "cpu_util": diag.get("cpu_util", 0.0),
            "bw_util": diag.get("bw_util", 0.0),

            # migration
            "migration_freq": diag.get("migration_freq", 0),
            "migration_cost": diag.get("migration_cost", 0.0),

            # availability
            "available": diag.get("available", 0),

            # complexity
            "ops_count": diag.get("ops_count", 0),

            # runtime
            "solve_time_s": decision.get("_solve_time_s", 0.0),
        }

        self.slots.append(rec)

        # 累加
        total_delay = (
            rec["delay_queue"] +
            rec["delay_tx"] +
            rec["delay_proc"]
        )
        if total_delay > 0:
            self.delay_samples.append(total_delay)

        self.total_tasks += rec["arrival_tasks"]
        self.completed_tasks += rec["completed_tasks"]
        self.dropped_tasks += rec["dropped_tasks"]

        self.handover_count += rec["handover"]

        self.energy_tx_total += rec["energy_tx"]
        self.energy_srv_total += rec["energy_srv"]
        self.effective_bits_total += rec["effective_bits"]

        self.cpu_util_samples.append(rec["cpu_util"])
        self.bw_util_samples.append(rec["bw_util"])

        self.migration_freq_total += rec["migration_freq"]
        self.migration_cost_total += rec["migration_cost"]

        self.availability_samples.append(rec["available"])

        self.ops_samples.append(rec["ops_count"])

        self.solve_times.append(rec["solve_time_s"])

    # ---------------------------------------------------------------------
    # 汇总输出全部 30+ 指标
    # ---------------------------------------------------------------------
    def summarize(self) -> Dict[str, Any]:
        S = {}
        N = len(self.slots)
        if N == 0:
            return S

        # ------------------------------
        # 1. 基本统计
        # ------------------------------
        S["slots"] = N

        # cost
        costs = np.array([s["total_cost"] for s in self.slots], float)
        S["C_mean"] = float(costs.mean())
        S["C_std"] = float(costs.std())

        # energy
        S["E_tx_total"] = self.energy_tx_total
        S["E_srv_total"] = self.energy_srv_total
        S["E_mean"] = self.energy_tx_total + self.energy_srv_total

        # throughput
        S["effective_bits_total"] = self.effective_bits_total
        S["Energy_per_bit"] = (self.energy_tx_total + self.energy_srv_total) / max(1e-9, self.effective_bits_total)

        S["Energy_per_task"] = (self.energy_tx_total + self.energy_srv_total) / max(1, self.completed_tasks)
        S["EE_proxy_bit_per_J"] = self.effective_bits_total / max(1e-9, (self.energy_tx_total + self.energy_srv_total))

        # ------------------------------
        # 2. 延迟
        # ------------------------------
        if self.delay_samples:
            d = np.array(self.delay_samples)
            S["avg_delay_mean"] = float(d.mean())
            S["avg_delay_median"] = float(np.median(d))
            S["delay_95"] = float(np.percentile(d, 95))
            S["delay_99"] = float(np.percentile(d, 99))
            S["SLA_violation_rate"] = float(np.mean(d > 0.2))
        else:
            S["avg_delay_mean"] = 0.0
            S["avg_delay_median"] = 0.0
            S["delay_95"] = 0.0
            S["delay_99"] = 0.0
            S["SLA_violation_rate"] = 0.0

        # queue
        queues = [np.mean(s["queue_len"]) for s in self.slots]
        S["Avg_queue"] = float(np.mean(queues))

        # ------------------------------
        # 3. 完成率
        # ------------------------------
        S["Acc_rate"] = self.completed_tasks / max(1, self.total_tasks)
        S["Drop_rate"] = self.dropped_tasks / max(1, self.total_tasks)

        # ------------------------------
        # 4. Mobility
        # ------------------------------
        S["Handover_count_total"] = self.handover_count

        # ------------------------------
        # 5. System Availability
        # ------------------------------
        S["Availability"] = float(np.mean(self.availability_samples))

        # ------------------------------
        # 6. Resource Utilization
        # ------------------------------
        S["CPU_util_mean"] = float(np.mean(self.cpu_util_samples))
        S["BW_util_mean"] = float(np.mean(self.bw_util_samples))

        # ------------------------------
        # 7. Migration statistics
        # ------------------------------
        S["Migration_freq_total"] = self.migration_freq_total
        S["Migration_cost_total"] = self.migration_cost_total

        # ------------------------------
        # 8. Complexity proxy
        # ------------------------------
        S["Ops_mean"] = float(np.mean(self.ops_samples))
        S["Ops_total"] = float(sum(self.ops_samples))

        # ------------------------------
        # 9. Run-time performance
        # ------------------------------
        S["DecisionTime_ms_mean"] = float(np.mean(self.solve_times)) * 1000
        S["DecisionTime_ms_max"] = float(np.max(self.solve_times)) * 1000

        return S

    # ---------------------------------------------------------------------
    # 保存 slot 数据 CSV
    # ---------------------------------------------------------------------
    def save_csv(self, filename="metrics_raw.csv"):
        if not self.slots:
            return None
        path = os.path.join(self.out_dir, filename)

        keys = sorted(set().union(*[s.keys() for s in self.slots]))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for s in self.slots:
                clean = {
                    k: (str(v) if isinstance(v, (list, dict)) else v)
                    for k, v in s.items()
                }
                w.writerow(clean)
        return path