# solvers/OLMA_Solver_perfect.py
"""
Fixed (minimal-change) OLMA solver implementation aligned to the provided environment units.
- Keeps original interfaces and metrics dict identical.
- Minimal changes: fix unit mismatches, stable initialization, robust heuristic SP2.
"""
import numpy as np
import math
from typing import Dict, Any
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings("default", category=UserWarning)

# try to import cvxpy; if not available we will fallback to heuristics
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except Exception:
    CVXPY_AVAILABLE = False

EPS = 1e-9

class OLMA_Solver:
    def __init__(self, env_config: Dict[str, Any], cfg: Dict[str, Any]):
        self.env_config = env_config or {}
        self.solver_config = cfg or {}
        self.V = float(self.solver_config.get('V', 1.0))
        self.bcd_max_iter = int(self.solver_config.get('bcd_max_iter', 6))
        self.sca_max_iter = int(self.solver_config.get('sca_max_iter', 6))
        self.eps = float(self.solver_config.get('eps', 1e-4))

        # solver preference for cvxpy (kept but heuristic path preferred)
        self.cvx_solver = None
        if CVXPY_AVAILABLE:
            installed = cp.installed_solvers()
            if 'ECOS' in installed:
                self.cvx_solver = cp.ECOS
            elif 'OSQP' in installed:
                self.cvx_solver = cp.OSQP
            elif 'SCS' in installed:
                self.cvx_solver = cp.SCS

    # ---------------------------
    # Helper: server dynamic energy (match environment: kappa * f^2 * Delta_t)
    # ---------------------------
    def server_dynamic_energy(self, kappa_j: float, f_vj: float, Delta_t: float) -> float:
        # environment uses kappa * f^2 * Delta_t (see environment.step)
        return float(kappa_j) * (f_vj ** 2) * Delta_t

    # ---------------------------
    # Main solve()
    # ---------------------------
    def solve(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        # parse state
        V_set = system_state.get('V_set', [])
        Nv = max(1, len(V_set))
        J_set = system_state.get('J_set', [])
        Jn = max(1, len(J_set))
        cloud_index = system_state.get('cloud_index', None)

        params = system_state.get('params', {})
        Delta_t = float(params.get('Delta_t', 1.0))
        sigma2 = float(params.get('sigma2', 1e-9))

        # Defaults, note: Bup_max expected in MHz (environment uses MHz)
        Pmax_v = np.array(params.get('Pmax_v', [1.0]*Nv), dtype=float)
        Bup_max = np.array(params.get('Bup_max', [10.0]*Jn), dtype=float)  # MHz
        Fmax_j = np.array(params.get('Fmax_j', [2e9]*Jn), dtype=float)     # Hz
        kappa_j = np.array(params.get('kappa_j', [1e-27]*Jn), dtype=float)

        tasks = system_state.get('tasks', [])
        if not tasks or len(tasks) < Nv:
            # safe fallback (Din in MB as environment uses MB)
            tasks = [{'Din': 0.0, 'Cv': 0.0, 'kv': 0} for _ in range(Nv)]

        prev_assignment = system_state.get('prev_assignment', [-1]*Nv)
        Qjk_callable = system_state.get('Qjk_func', None)
        def Qjk(j,k):
            if callable(Qjk_callable):
                try:
                    return float(Qjk_callable(j,k))
                except Exception:
                    return 0.0
            return 0.0

        g = np.array(system_state.get('g', np.ones((Nv, Jn))), dtype=float)
        # defensive shape fix
        if g.shape != (Nv, Jn):
            g = np.ones((Nv, Jn), dtype=float)

        # initialize decision vars (feasible and in same units as environment)
        # power in Watts, bandwidth in MHz, frequency in Hz
        p = np.minimum(Pmax_v, 0.5 * Pmax_v + 0.1)  # reasonably sized transmit power
        B = np.zeros((Nv, Jn), float)               # MHz
        for v in range(Nv):
            B[v, np.argmax(g[v])] = min( Bup_max.mean() / max(1, Nv), Bup_max.mean() * 0.5 )
        # Initialize F with a practical per-vehicle share of server CPU (Hz)
        F = np.zeros((Nv, Jn), float)
        for j in range(Jn):
            per_v = max(1, Nv)
            share = min(Fmax_j[j] / per_v, Fmax_j[j] * 0.01)
            for v in range(Nv):
                F[v, j] = share

        # assign to best server initially
        assignment = np.argmax(g, axis=1).astype(int)

        # metrics placeholders
        metrics = {}

        # ---------------------------
        # BCD loop
        # ---------------------------
        for bcd_it in range(self.bcd_max_iter):

            # ---------------------------
            # 1) interference computation
            # ---------------------------
            Im = np.zeros(Jn, float)
            for v in range(Nv):
                a_v = int(assignment[v])
                for j in range(Jn):
                    if j != a_v:
                        Im[j] += p[v] * g[v, j]

            # ---------------------------
            # 2) rate computation (units aligned)
            # B in MHz → Hz: B * 1e6
            # ---------------------------
            rate_vj = np.zeros((Nv, Jn), float)
            for v in range(Nv):
                for j in range(Jn):
                    Bj_Hz = float(max(B[v, j], 1e-9)) * 1e6   # convert to Hz
                    denom = Im[j] + sigma2
                    gamma = (p[v] * g[v, j]) / (denom + EPS)
                    rate_vj[v, j] = Bj_Hz * math.log2(1 + max(gamma, 1e-12))  # bits/s

            # ---------------------------
            # 3) build omega matrix (SP1)
            # ---------------------------
            omega = np.zeros((Nv, Jn), float)
            for v in range(Nv):
                Din_MB = float(tasks[v].get('Din', 0.0))
                Din_bits = Din_MB * 8e6  # MB → bits (environment uses)
                Cv = float(tasks[v].get('Cv', 0.0))
                kv = int(tasks[v].get('kv', 0))

                for j in range(Jn):

                    R_vj = rate_vj[v, j] + EPS
                    # Tx energy = p * (Din_bits / rate)
                    if R_vj > 1e-12:
                        E_tx = p[v] * (Din_bits / R_vj)
                    else:
                        E_tx = 1e9

                    # server dynamic energy (kappa f^2 Δt)
                    f_vj = float(F[v, j])
                    E_srv = self.server_dynamic_energy(kappa_j[j], f_vj, Delta_t)

                    Cbar = E_tx + E_srv

                    # queue/conflict term
                    qjk = Qjk(j, kv)

                    # service benefit: phi * sum f * Δt
                    try:
                        phi_jk = float(params.get("phi", [[1.0]*1 for _ in range(Jn)])[j][kv])
                    except Exception:
                        phi_jk = 1.0

                    f_sum_j = float(np.sum(F[:, j]))
                    service_benefit = qjk * phi_jk * f_sum_j * Delta_t

                    # handover penalty
                    prev_j = int(prev_assignment[v]) if v < len(prev_assignment) else -1
                    hv = 1 if (prev_j != -1 and prev_j != j) else 0
                    wH = params.get('wH', 0.0)

                    omega[v, j] = self.V * Cbar + qjk * Cv - service_benefit + wH * hv

                    # infeasible: rate too low to upload Din within Δt
                    if R_vj * Delta_t < Din_bits:
                        omega[v, j] += 1e8

            metrics["omega"] = omega.copy()
            metrics["Im"] = Im.copy()
            metrics["rate_vj"] = rate_vj.copy()

            # ---------------------------
            # 4) SP1 Hungarian matching
            # ---------------------------
            try:
                row_ind, col_ind = linear_sum_assignment(omega)
                assignment_new = np.full(Nv, -1, int)
                assignment_new[row_ind] = col_ind
            except Exception:
                assignment_new = np.argmin(omega, axis=1)

            # ---------------------------
            # 5) SP2 continuous update (SAFE heuristics)
            # ---------------------------
            # 说明：不大改原结构，但让 SP2 一定产生非零可用的 p/B/F
            # 与 environment 中能量模型一致。

            for v in range(Nv):
                j = int(assignment_new[v])

                # power: 不为零
                p[v] = min(Pmax_v[v], max(0.2 * Pmax_v[v], 0.05))

                # bandwidth: 给该 server 一个均匀份额 (MHz)
                B[v, :] = 0.0
                B[v, j] = max(1e-3, Bup_max[j] / max(1, Nv))

                # freq: 给 server 均匀分配 Hz
                F[v, :] = 0.0
                F[v, j] = min(Fmax_j[j] / max(1, Nv), Fmax_j[j] * 0.02)

            # update assignment
            assignment_old = assignment.copy()
            assignment = assignment_new.copy()

            # stopping condition
            if np.array_equal(assignment_old, assignment):
                break

        # ---------------------------
        # After BCD, compute full metrics
        # ---------------------------
        # Ajk: total cycles arrival at each server (cycles)
        Ajk = {}
        for j in range(Jn):
            Ajk[j] = 0.0
        for v in range(Nv):
            j = int(assignment[v])
            if 0 <= j < Jn:
                Ajk[j] += float(tasks[v].get('Cv', 0.0))

        # Sjk per paper: phi_jk * sum_v a_vj f_vj * Delta_t (cycles served)
        Sjk = {}
        for j in range(Jn):
            f_sum = float(np.sum(F[:, j]))  # Hz sum
            for v in range(Nv):
                kv = int(tasks[v].get('kv', 0))
                phi_jk = 1.0
                try:
                    phi_jk = float(params.get('phi', [[1.0]*1 for _ in range(Jn)])[j][kv])
                except Exception:
                    phi_jk = 1.0
                Sjk[(j, kv)] = phi_jk * f_sum * Delta_t

        # compute final rate, Etx, Esrv and interference
        rate_final = np.zeros((Nv, Jn), float)
        E_tx_vj = np.zeros((Nv, Jn), float)
        E_srv_vj = np.zeros((Nv, Jn), float)
        Im_final = np.zeros(Jn, float)
        for vp in range(Nv):
            a_vp = int(assignment[vp])
            for m in range(Jn):
                if a_vp != m:
                    Im_final[m] += p[vp] * g[vp, m]

        for v in range(Nv):
            Din_MB = float(tasks[v].get('Din', 0.0))
            Din_bits = Din_MB * 8e6
            for j in range(Jn):
                Bj_Hz = max(B[v, j], 1e-9) * 1e6
                denom = Im_final[j] + sigma2 + EPS
                gamma = (p[v] * g[v, j]) / denom
                rate_final[v, j] = max(Bj_Hz * math.log2(1.0 + max(gamma, 1e-12)), 0.0)
                if rate_final[v, j] * Delta_t > EPS:
                    E_tx_vj[v, j] = p[v] * (Din_bits / (rate_final[v, j] * Delta_t))
                else:
                    E_tx_vj[v, j] = 0.0
                Cv = float(tasks[v].get('Cv', 0.0))
                E_srv_vj[v, j] = self.server_dynamic_energy(kappa_j[j], F[v, j], Delta_t)

        # E_ctrl (control plane) basic proxy
        E_ctrl = float(params.get('E_ctrl_base', 0.0))

        # totals
        Ev_total = 0.0
        for v in range(Nv):
            j = int(assignment[v])
            if 0 <= j < Jn:
                Ev_total += E_tx_vj[v, j]
        Ej_total = float(np.sum(E_srv_vj))
        E_sys = Ev_total + Ej_total + E_ctrl

        # cloud usage and handover
        cloud_usage = 0.0
        if cloud_index is not None:
            for v in range(Nv):
                if int(assignment[v]) == cloud_index:
                    # note: keep unit consistent (Din in MB * wC)
                    cloud_usage += float(tasks[v].get('Din', 0.0)) * float(params.get('wC', 0.0))

        handover_count = 0
        for v in range(Nv):
            prev_j = int(prev_assignment[v]) if v < len(prev_assignment) else -1
            if prev_j != -1 and prev_j != int(assignment[v]):
                handover_count += 1

        # combined system cost
        wE = float(params.get('wE', 1.0))
        wH = float(params.get('wH', 0.0))
        C_sys = wE * E_sys + cloud_usage + wH * handover_count

        # Lyapunov term (sum_jk Qjk (A_jk - S_jk))
        Lyap_term = 0.0
        for j in range(Jn):
            Aj = 0.0
            for v in range(Nv):
                if int(assignment[v]) == j:
                    Aj += float(tasks[v].get('Cv', 0.0))
            Sj = float(np.sum(F[:, j])) * Delta_t
            Qval = Qjk(j, 0)
            Lyap_term += Qval * (Aj - Sj)

        metrics = {
            'assignment': assignment.tolist(),
            'power': p.tolist(),
            'bandwidth': B.tolist(),   # MHz
            'freq': F.tolist(),        # Hz
            'Ajk': Ajk,
            'Sjk': Sjk,
            'E_tx_vj': E_tx_vj,
            'E_srv_vj': E_srv_vj,
            'E_sys': E_sys,
            'C_sys': C_sys,
            'handover_count': handover_count,
            'cloud_usage': cloud_usage,
            'Im': Im_final,
            'rate_vj': rate_final,
            'omega': metrics.get('omega', None),
            'Lyapunov_term': Lyap_term
        }

        return metrics