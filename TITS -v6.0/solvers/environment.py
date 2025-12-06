# ================================================================
#  FULL PAPER-STYLE VEC ENVIRONMENT (TITS-Compatible)
#  Strictly compatible with OLMA_Solver and run_plot
#  Author: ChatGPT (customized for your project)
# ================================================================

import numpy as np
from typing import Dict, Any, List
import math

# -----------------------------
# UNIT CONSTANTS
# -----------------------------
BITS_PER_MB = 8 * 10**6        # MB -> bits
SECONDS_PER_SLOT = 1           # 默认 Δt = 1 秒，可改成环境变量

class VEC_Environment:
    """
    Full Version VEC Environment for OLMA (replicates TITS paper structure)
    Supports all solver inputs and all performance indicators.
    """

    # -------------------------------------------------------------
    # INIT
    # -------------------------------------------------------------
    def __init__(self, cfg: Dict[str, Any]):
        cfg = cfg or {}
        self.cfg = cfg
        self.RNG = np.random.default_rng(seed=cfg.get("seed", 42))

        # ---------------------- System Size -----------------------
        self.num_servers = int(cfg.get("num_servers", 3))
        self.num_vehicles = int(cfg.get("num_vehicles", 8))

        # ---------------------- Time Slot -------------------------
        self.Delta_t = float(cfg.get("Delta_t", 1.0))

        # ---------------------- Vehicle Mobility ------------------
        self.max_speed = float(cfg.get("max_speed", 15.0))
        self.area_size = float(cfg.get("area_size", 600.0))

        # ---------------------- Comm Range ------------------------
        self.comm_range = float(cfg.get("comm_range", 200.0))

        # ---------------------- Radio Parameters ------------------
        self.sigma2 = float(cfg.get("sigma2", 1e-9))
        self.path_loss_exp = float(cfg.get("path_loss_exp", 2.1))
        self.fading_std = float(cfg.get("fading_std", 0.1))

        # ---------------------- Resource Capacity -----------------
        self.Pmax = np.array(cfg.get("Pmax_v", [2.0] * self.num_vehicles))
        self.Bmax = np.array(cfg.get("Bmax", [12.0] * self.num_servers))
        self.Fmax = np.array(cfg.get("Fmax_j", [2.5e9] * self.num_servers))
        self.kappa_j = np.array(cfg.get("kappa_j", [1e-27] * self.num_servers))

        # ---------------------- Task Model ------------------------
        self.Din_low = float(cfg.get("Din_low", 0.2))
        self.Din_high = float(cfg.get("Din_high", 1.2))
        self.Cv_base = float(cfg.get("Cv_base", 2e5))

        # ---------------------- Handover & Migration --------------
        self.handover_cost = float(cfg.get("handover_cost", 0.3))

        # ---------------------- Queue Model -----------------------
        self.queue_j = np.zeros(self.num_servers)
        self.prev_assignment = [-1] * self.num_vehicles

        # ---------------------- Server Positions ------------------
        self.server_positions = self._init_server_positions()

        # ---------------------- Runtime States --------------------
        self.vehicle_positions = None
        self.vehicle_dirs = None
        self.external_tasks = None

    # -------------------------------------------------------------
    #  SERVER POSITIONS (fixed layout)
    # -------------------------------------------------------------
    def _init_server_positions(self):
        coords = []
        for m in range(self.num_servers):
            x = self.area_size * (m + 1) / (self.num_servers + 1)
            y = self.area_size * 0.45
            coords.append([x, y])
        return np.array(coords)

    # -------------------------------------------------------------
    #  RESET ENVIRONMENT FOR EPISODE
    # -------------------------------------------------------------
    def reset(self):
        self.vehicle_positions = self.RNG.uniform(0, self.area_size,
                                                  size=(self.num_vehicles, 2))
        dirs = self.RNG.uniform(-1, 1, size=(self.num_vehicles, 2))
        self.vehicle_dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

        self.queue_j[:] = 0
        self.prev_assignment = [-1] * self.num_vehicles

    # --------------------------------------------
    # CPU energy model: E = kappa * f^2 * Delta_t
    # --------------------------------------------
    def server_energy(self, kappa_j: float, f: float, Delta_t: float) -> float:
        return float(kappa_j) * (float(f) ** 2) * float(Delta_t)

    # -------------------------------------------------------------
    #  UPDATE VEHICLE POSITIONS
    # -------------------------------------------------------------
    def _update_vehicle_positions(self):
        for v in range(self.num_vehicles):
            speed = self.max_speed * (0.5 + 0.5 * self.RNG.random())
            move = self.vehicle_dirs[v] * speed
            self.vehicle_positions[v] += move

            # reflect at boundaries
            for i in range(2):
                if self.vehicle_positions[v, i] < 0 or \
                   self.vehicle_positions[v, i] > self.area_size:
                    self.vehicle_dirs[v, i] *= -1
                    self.vehicle_positions[v, i] = np.clip(
                        self.vehicle_positions[v, i],
                        0, self.area_size
                    )

    # -------------------------------------------------------------
    #  CHANNEL GAIN MODEL (Distance + Shadowing + Fading)
    # -------------------------------------------------------------
    def _channel_gain(self, v_idx: int, j_idx: int):
        pos_v = self.vehicle_positions[v_idx]
        pos_s = self.server_positions[j_idx]

        d = np.linalg.norm(pos_v - pos_s)
        if d > self.comm_range:
            return 1e-9

        # path loss
        path_loss = (1.0 + d ** self.path_loss_exp)

        # log-normal shadowing
        shadow = 10 ** (self.RNG.normal(0, self.fading_std) / 10)

        # final channel gain
        g = 1.0 / path_loss * shadow
        return max(g, 1e-9)

    # -------------------------------------------------------------
    #  GENERATE TASK FOR A VEHICLE (STRICT PAPER STYLE)
    # -------------------------------------------------------------
    def _generate_task(self):
        Din = float(self.RNG.uniform(self.Din_low, self.Din_high))     # MB
        Cv = float(self.Cv_base * self.RNG.uniform(0.9, 1.1))          # cycles
        kv = 0                                                         # not used but kept

        return {
            "Din": Din,
            "Cv": Cv,
            "kv": kv
        }

    def load_external_tasks(self, tasks_df):
        """
        加载外部真实/固定任务数据:
        任务数据格式必须至少包含: Din, Cv (可选 kv)
        """
        self.external_tasks = []

        for _, row in tasks_df.iterrows():
            self.external_tasks.append({
                "Din": float(row["Din"]),
                "Cv": float(row.get("Cv", self.Cv_base)),  # 若没有 Cv 则给默认
                "kv": float(row.get("kv", 0))
            })

    def export_current_tasks(self):
        """
        将当前任务（内部随机生成）导出为 DataFrame
        run_plot 用来做公用随机任务集
        """
        import pandas as pd

        # 必须现调用一次 get_state() 才有 self.latest_tasks
        if not hasattr(self, "latest_tasks"):
            raise RuntimeError("请先调用 get_state() 再 export 当前任务！")

        data = {
            "Din": [t["Din"] for t in self.latest_tasks],
            "Cv": [t["Cv"] for t in self.latest_tasks],
            "kv": [t["kv"] for t in self.latest_tasks],
        }
        return pd.DataFrame(data)

    # -------------------------------------------------------------
    #  GET STATE (strictly compatible with OLMA solver)
    # -------------------------------------------------------------
    def get_state(self):
        Nv = self.num_vehicles
        Jn = self.num_servers

        # update mobility first
        self._update_vehicle_positions()

        # -------------------------
        # 1. Generate channel gain
        # -------------------------
        g = np.zeros((Nv, Jn))
        for v in range(Nv):
            for j in range(Jn):
                g[v, j] = self._channel_gain(v, j)

        # -------------------------
        # 2. Generate new tasks
        # -------------------------
        tasks = []
        # --- 若存在外部任务，则使用外部任务 ---
        if self.external_tasks is not None:
            if len(self.external_tasks) < Nv:
                raise ValueError("外部任务数量不足 Nv")
            tasks = self.external_tasks[:Nv]

        else:
            # 否则生成内部随机任务
            tasks = [self._generate_task() for _ in range(Nv)]

        # 保存一份给 export_current_tasks()
        self.latest_tasks = tasks

        # -------------------------
        # 3. State required by solver
        # -------------------------
        state = {
            "V_set": list(range(Nv)),
            "J_set": list(range(Jn)),
            "g": g,
            "tasks": tasks,

            "cloud_index": None,   # reserved
            "params": {
                "Delta_t": self.Delta_t,
                "sigma2": self.sigma2
            },

            # queue lookup (paper uses Q_{j,k})
            "Qjk_func": lambda j, k: self.queue_j[j],

            "prev_assignment": self.prev_assignment
        }

        return state

    # -------------------------------------------------------------
    #  HELPER: compute rate given power, bw (MHz) and channel g, interference I
    # -------------------------------------------------------------
    def _compute_rate(self, p_v, bw_MHz, g_vj, I_j, sigma2):
        # bw_MHz -> Hz
        Bw = max(1e-12, float(bw_MHz)) * 1e6
        gamma = (float(p_v) * float(g_vj)) / (float(I_j) + float(sigma2) + 1e-18)
        gamma = max(gamma, 0.0)
        rate = Bw * math.log2(1.0 + gamma)  # bits/s
        return rate, gamma

    # -------------------------------------------------------------
    #  STEP: apply decision for one slot and return diagnostics
    #  decision must contain:
    #   - assignment: list of length Nv
    #   - power: list or array length Nv (W)
    #   - bandwidth: Nv x Jn matrix (MHz)
    #   - freq: Nv x Jn matrix (Hz)
    # -------------------------------------------------------------
    def step(self, decision: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        Nv = self.num_vehicles
        Jn = self.num_servers

        # Unpack decision safely (support list-of-lists or numpy arrays)
        assign = np.array(decision.get("assignment", [0]*Nv), dtype=int)
        power = np.array(decision.get("power", [0.0]*Nv), dtype=float).flatten()

        # bandwidth and freq may be lists-of-lists or nested arrays
        bw = np.array(decision.get("bandwidth", np.zeros((Nv, Jn))), dtype=float)
        freq = np.array(decision.get("freq", np.zeros((Nv, Jn))), dtype=float)

        # Cap shapes
        if bw.shape != (Nv, Jn):
            try:
                bw = np.reshape(bw, (Nv, Jn))
            except Exception:
                bw = np.zeros((Nv, Jn), dtype=float)
        if freq.shape != (Nv, Jn):
            try:
                freq = np.reshape(freq, (Nv, Jn))
            except Exception:
                freq = np.zeros((Nv, Jn), dtype=float)

        # Unpack state-provided items (g, tasks)
        g = np.array(state.get("g", np.ones((Nv, Jn))), dtype=float)
        tasks = state.get("tasks", [{"Din": 0.0, "Cv": 0.0, "kv": 0} for _ in range(Nv)])
        params = state.get("params", {})

        Delta = float(params.get("Delta_t", self.Delta_t))
        sigma2 = float(params.get("sigma2", self.sigma2))

        # Diagnostics
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
        migration_cost = 0.0

        used_cpu_by_server = np.zeros(Jn, dtype=float)
        used_bw_per_server = np.zeros(Jn, dtype=float)

        ops_count = 0

        # Precompute interference per server: sum p[v]*g[v,m] for v assigned to other servers
        Im = np.zeros(Jn, dtype=float)
        for vp in range(Nv):
            a_vp = int(assign[vp])
            for m in range(Jn):
                if m != a_vp:
                    Im[m] += float(power[vp]) * float(g[vp, m])

        # Iterate each vehicle
        for v in range(Nv):
            j = int(assign[v])
            # bounds check
            if j < 0 or j >= Jn:
                # invalid assignment => drop
                dropped_tasks += 1
                continue

            # Handover
            prev_j = int(self.prev_assignment[v]) if v < len(self.prev_assignment) else -1
            if prev_j != -1 and prev_j != j:
                handover_cnt += 1

            # Task params
            Din_MB = float(tasks[v].get("Din", 0.0))
            Din_bits = Din_MB * BITS_PER_MB  # MB -> bits
            Cv = float(tasks[v].get("Cv", 0.0))  # cycles
            ops_count += int(max(0, Cv))

            # Compute instantaneous interference seen at server j (from vehicles assigned elsewhere)
            Ij = float(Im[j])

            # Transmission
            p_v = float(power[v]) if v < len(power) else 0.0
            bw_vj_MHz = float(bw[v, j]) if v < bw.shape[0] else 0.0

            used_bw_per_server[j] += bw_vj_MHz

            rate, gamma = self._compute_rate(p_v, bw_vj_MHz, g[v, j], Ij, sigma2)  # bits/s

            # If rate is too small, drop task (cannot upload)
            if rate < 1e-9 or Din_bits <= 0.0:
                dropped_tasks += 1
                continue

            # Transmission time (s) and energy (J)
            tx_time = float(Din_bits) / max(rate, 1e-12)
            tx_energy = float(p_v) * tx_time
            total_tx_energy += tx_energy
            effective_bits_total += float(Din_bits)
            delay_tx_total += tx_time

            # Queue and compute at server j
            # Queue before arrival is current queue_j[j] (cycles)
            Q_before = float(self.queue_j[j])

            # incoming cycles added to queue (we model arrival immediately)
            self.queue_j[j] += Cv

            # CPU frequency allocation for this vehicle at server j
            fj = float(freq[v, j]) if v < freq.shape[0] else 0.0
            # Enforce minimal positive freq to avoid division by zero
            fj = max(1e-6, fj)

            # Served cycles in this slot by server j (aggregate of all vehicles using freq at that server)
            served_cycles = fj * Delta
            used_cpu_by_server[j] += served_cycles

            # service: remove from queue as much as served_cycles allows
            service = min(self.queue_j[j], served_cycles)
            self.queue_j[j] -= service

            # compute processing time for the vehicle's task (assuming serial execution of its Cv on allocated fj)
            proc_time = float(Cv) / max(fj, 1e-12)
            delay_proc_total += proc_time

            # approximate queue delay experienced by this task: queue_before / served_rate (cycles per slot)
            if served_cycles > 0:
                delay_queue = Q_before / served_cycles
            else:
                delay_queue = float('inf')
            delay_queue_total += delay_queue

            # CPU energy consumption for serving (κ * f^2 * Δt)
            kappa_val = float(self.kappa_j[j]) if j < len(self.kappa_j) else float(self.kappa_j.mean())
            cpu_energy = self.server_energy(kappa_val, fj, Delta)
            total_srv_energy += cpu_energy

            # Completed task counting heuristic:
            # If service >= Cv (i.e., we were able to compute the task cycles this slot), mark completed
            if service >= Cv:
                completed_tasks += 1
            else:
                # partial service -- mark as dropped if queue grows too large (paper uses queue stability, here heuristic)
                # We'll not increment dropped here; unfinished tasks stay in queue for next slots
                pass

        # finalize dropped tasks: if queue backlog or upload failures caused deficits, ensure consistency
        # simple heuristic: arrivals - completed = dropped if queue hasn't absorbed them
        dropped_tasks += max(0, arrival_tasks - completed_tasks - int(np.sum(self.queue_j) > 0)*0)

        # update prev_assignment
        self.prev_assignment = list(assign)

        # Resource utilization metrics
        cpu_util_per_server = np.zeros(Jn, dtype=float)
        for m in range(Jn):
            cap_cycles = max(1e-12, float(self.Fmax[m]) * Delta)  # Fmax in Hz => cycles per second approximation
            cpu_util_per_server[m] = min(1.0, used_cpu_by_server[m] / cap_cycles) if cap_cycles > 0 else 0.0
        cpu_util = float(np.mean(cpu_util_per_server))

        bw_util_per_server = np.zeros(Jn, dtype=float)
        for m in range(Jn):
            cap_bw = max(1e-12, float(self.Bmax[m]))  # MHz
            bw_util_per_server[m] = min(1.0, used_bw_per_server[m] / cap_bw) if cap_bw > 0 else 0.0
        bw_util = float(np.mean(bw_util_per_server))

        migration_freq = int(handover_cnt)
        migration_cost = float(migration_freq) * float(self.handover_cost)

        available = 1 if completed_tasks >= max(1, arrival_tasks) else 0

        # Total cost definition consistent with your run_plot: energy-based cost
        total_cost = float(total_tx_energy + total_srv_energy + migration_cost)

        # ops_count as proxy for computation done
        ops_count = int(ops_count)

        # Prepare diagnostics dictionary (keys used by run_plot/self-tests)
        diagnostics = {
            "total_cost": total_cost,
            "queue_lengths": self.queue_j.copy(),
            "arrival_tasks": int(arrival_tasks),
            "completed_tasks": int(completed_tasks),
            "dropped_tasks": int(dropped_tasks),
            "delay_queue": float(delay_queue_total),
            "delay_tx": float(delay_tx_total),
            "delay_proc": float(delay_proc_total),
            "delay_backhaul": 0.0,
            "energy_tx": float(total_tx_energy),
            "energy_srv": float(total_srv_energy),
            "effective_bits": float(effective_bits_total),
            "handover": int(handover_cnt),
            "cpu_util": float(cpu_util),
            "bw_util": float(bw_util),
            "migration_freq": int(migration_freq),
            "migration_cost": float(migration_cost),
            "available": int(available),
            "ops_count": int(ops_count),
        }

        return diagnostics

    # -------------------------------------------------------------
    # END OF CLASS
    # -------------------------------------------------------------
