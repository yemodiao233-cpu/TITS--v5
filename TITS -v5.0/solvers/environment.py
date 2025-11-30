import numpy as np
from typing import Dict, Any, List


class VEC_Environment:
    """
    终极版 VEC 环境，用于完全复现 TITS 论文所需指标。

    输出支持所有 30+ paper 指标：
    - 成本 Cost
    - Transmission / Server Energy
    - 队列 Queue
    - 服务量 Completed / Dropped
    - 延迟 Delay (queue + radio + compute + backhaul)
    - 有效吞吐 Effective bits
    - CPU utilization
    - Bandwidth utilization
    - Migration frequency + cost
    - Service availability
    - Complexity (ops_count)
    - 等等……
    """

    def __init__(self, cfg: Dict[str, Any]):
        cfg = cfg or {}
        self.cfg = cfg

        # --------------------------
        # 基础规模参数
        # --------------------------
        self.num_servers = int(cfg.get('num_servers', 2))
        self.num_vehicles = int(cfg.get('num_vehicles', 3))

        # slot 时间
        self.Delta_t = float(cfg.get('Delta_t', 1.0))
        self.comm_range = float(cfg.get('comm_range', 150.0))

        # 上行功率上限（每车）
        self.Pmax = np.array(cfg.get('Pmax', [1.0] * self.num_vehicles), float)

        # 服务器总带宽（MHz）
        self.Bmax = np.array(cfg.get('Bmax', [10.0] * self.num_servers), float)

        # MEC CPU 最大频率（Hz）
        self.Fmax = np.array(cfg.get('Fmax', [2e9] * self.num_servers), float)

        # CPU 能耗 κ f^2
        self.kappa_j = np.array(cfg.get('kappa_j', [1e-27] * self.num_servers))

        # 噪声
        self.sigma2 = float(cfg.get('sigma2', 1e-9))

        # handover 代价（可配置）
        self.handover_cost = float(cfg.get("handover_cost", 0.1))

        # 服务器位置
        self.server_positions = self._init_servers()

        # 运行状态
        self.positions = None
        self.queue_j = np.zeros(self.num_servers, float)
        self.prev_assignment = [-1] * self.num_vehicles

    # ---------------------------------------------------------
    # 初始化 MEC 服务器位置
    # ---------------------------------------------------------
    def _init_servers(self):
        coords = []
        for m in range(self.num_servers):
            coords.append(np.array([200.0 * (m+1), 250.0]))  # 简单布局
        return np.vstack(coords)

    # ---------------------------------------------------------
    # 重置环境
    # ---------------------------------------------------------
    def reset(self):
        self.positions = np.random.rand(self.num_vehicles, 2) * 500.0
        self.queue_j = np.zeros(self.num_servers)
        self.prev_assignment = [-1] * self.num_vehicles

    # ---------------------------------------------------------
    # 信道增益（基于距离）
    # ---------------------------------------------------------
    def _channel_gain(self, v_idx: int, j_idx: int) -> float:
        d = np.linalg.norm(self.positions[v_idx] - self.server_positions[j_idx])
        if d > self.comm_range:
            return 1e-9
        return 1.0 / (1.0 + (d / 50.0)**2)

    # ---------------------------------------------------------
    # 获取当前系统状态
    # ---------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        g = np.zeros((self.num_vehicles, self.num_servers))
        for v in range(self.num_vehicles):
            for j in range(self.num_servers):
                g[v, j] = self._channel_gain(v, j)

        # 每车一个任务
        tasks = []
        for v in range(self.num_vehicles):
            Din = float(max(0.1, np.random.rand() * 0.5))  # MB
            tasks.append({
                "Din": Din,
                "Cv": float(2e5),  # cycles
                "kv": 0
            })

        return {
            "V_set": list(range(self.num_vehicles)),
            "J_set": list(range(self.num_servers)),
            "g": g,
            "tasks": tasks,
            "cloud_index": None,

            "params": {
                "Delta_t": self.Delta_t,
                "sigma2": self.sigma2,
            },

            # queue lookup
            "Qjk_func": lambda j, k: self.queue_j[j],
            "prev_assignment": self.prev_assignment
        }

    # ---------------------------------------------------------
    # 执行一个时隙
    # ---------------------------------------------------------
    def step(self, decision: Dict[str, Any], state: Dict[str, Any]):
        Nv = self.num_vehicles
        Jn = self.num_servers

        assign = decision["assignment"]
        power = np.array(decision["power"], float)
        bw = decision["bandwidth"]
        freq = decision["freq"]

        g = state["g"]
        tasks = state["tasks"]

        Delta = self.Delta_t

        # ------------------------------
        # 统计量初始化
        # ------------------------------
        total_tx_energy = 0.0
        total_srv_energy = 0.0
        effective_bits_total = 0.0

        delay_queue_total = 0.0
        delay_tx_total = 0.0
        delay_proc_total = 0.0

        arrival_tasks = Nv
        completed_tasks = 0
        dropped_tasks = 0

        handover_cnt = 0

        # CPU 利用率统计
        used_cpu_by_server = np.zeros(Jn, float)

        # 带宽利用率统计
        used_bw_per_server = np.zeros(Jn, float)

        # 复杂度 proxy
        ops_count = 0

        # ------------------------------
        # per-vehicle computation
        # ------------------------------
        for v in range(Nv):
            j = int(assign[v])

            # handover
            if self.prev_assignment[v] != -1 and self.prev_assignment[v] != j:
                handover_cnt += 1

            Din = tasks[v]["Din"]
            Cv = tasks[v]["Cv"]     # cycles
            ops_count += int(Cv)    # 复杂度 proxy

            # ------------------------------
            # 1) Transmission
            # ------------------------------
            gamma = (power[v] * g[v, j]) / (self.sigma2 + 1e-12)

            # bandwidth bw[v][j] — 单位 MHz → 转 Hz
            Bw = float(bw[v][j]) * 1e6

            used_bw_per_server[j] += float(bw[v][j])

            rate = Bw * np.log2(1 + gamma)  # bit/s

            if rate < 1e-9:
                dropped_tasks += 1
                continue

            bits = Din * 8 * (10**6)    # MB → bit

            tx_time = bits / max(rate, 1e-12)
            tx_energy = power[v] * tx_time

            total_tx_energy += tx_energy
            effective_bits_total += bits

            # ------------------------------
            # 2) Queue + Compute
            # ------------------------------
            Q_before = self.queue_j[j]

            # CPU frequency
            fj = float(freq[v][j])
            served_cycles = fj * Delta

            used_cpu_by_server[j] += served_cycles

            # queue arrivals
            self.queue_j[j] += Cv
            service = min(self.queue_j[j], served_cycles)

            self.queue_j[j] -= service

            # compute delay
            comp_time = Cv / max(fj, 1e-12)

            if served_cycles > 0:
                delay_queue = Q_before / served_cycles
            else:
                delay_queue = 0.0

            delay_proc_total += comp_time
            delay_queue_total += delay_queue
            delay_tx_total += tx_time

            # CPU energy κ f² t
            cpu_energy = self.kappa_j[j] * (fj**2) * Delta
            total_srv_energy += cpu_energy

            completed_tasks += 1

        dropped_tasks += max(0, arrival_tasks - completed_tasks)

        # 更新 assignment
        self.prev_assignment = list(assign)

        # ------------------------------
        # Resource Utilization
        # ------------------------------
        cpu_util_per_server = np.zeros(Jn)
        for m in range(Jn):
            cap = max(1e-12, self.Fmax[m] * Delta)
            cpu_util_per_server[m] = min(1.0, used_cpu_by_server[m] / cap)
        cpu_util = float(np.mean(cpu_util_per_server))

        bw_util_per_server = np.zeros(Jn)
        for m in range(Jn):
            cap = max(1e-12, self.Bmax[m])
            bw_util_per_server[m] = min(1.0, used_bw_per_server[m] / cap)
        bw_util = float(np.mean(bw_util_per_server))

        # ------------------------------
        # Migration cost
        # ------------------------------
        migration_freq = handover_cnt
        migration_cost = migration_freq * self.handover_cost

        # ------------------------------
        # Availability
        # ------------------------------
        available = 1 if completed_tasks >= max(1, arrival_tasks) else 0

        # ------------------------------
        # 返回 diagnostics
        # ------------------------------
        return {
            # cost
            "total_cost": total_tx_energy + total_srv_energy,

            # queue
            "queue_lengths": self.queue_j.copy(),

            # tasks
            "arrival_tasks": arrival_tasks,
            "completed_tasks": completed_tasks,
            "dropped_tasks": dropped_tasks,

            # delay components
            "delay_queue": delay_queue_total,
            "delay_tx": delay_tx_total,
            "delay_proc": delay_proc_total,
            "delay_backhaul": 0.0,

            # energy
            "energy_tx": total_tx_energy,
            "energy_srv": total_srv_energy,

            # throughput
            "effective_bits": effective_bits_total,

            # mobility
            "handover": handover_cnt,

            # NEW — 资源利用
            "cpu_util": cpu_util,
            "bw_util": bw_util,

            # NEW — 迁移
            "migration_freq": migration_freq,
            "migration_cost": migration_cost,

            # NEW — 系统可用度
            "available": available,

            # NEW — 复杂度 proxy
            "ops_count": ops_count,
        }